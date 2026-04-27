"""Gradio Web Demo for WildDet3D (5-mode).

Supports 5 prompt modes:
- Text: Enter text like "car.person.traffic light" (one-to-many)
- Visual: Click box on image, text="visual" (one-to-many)
- Visual+Label: Click box + category label (one-to-many)
- Geometry: Click box on image, text="geometric" (one-to-one)
- Geometry+Label: Click box + category label (one-to-one)
- Point: Click on image to select point

Requirements:
    pip install gradio>=5.0.0

Usage:
    python demo/huggingface/app.py

Then open http://localhost:7860 in browser.
"""

import os
import sys
from pathlib import Path

# Add paths: support both local dev and HuggingFace Space.
# Local dev:  demo/huggingface/app.py -> repo root = ../../
# HF Space:   wilddet3d/ is bundled in the same directory as app.py
_this_dir = Path(__file__).resolve().parent
if (_this_dir / "wilddet3d").exists():
    # HuggingFace Space: everything bundled next to app.py
    sys.path.insert(0, str(_this_dir))
else:
    # Local dev: repo root is two levels up
    repo_root = _this_dir.parent.parent
    sys.path.insert(0, str(repo_root))

try:
    import spaces
except ImportError:
    # Local run (not on HF Spaces) - create dummy decorator
    class _Spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _Spaces()

import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image

import json

from wilddet3d.inference import build_model, WildDet3DPredictor
from wilddet3d.preprocessing import preprocess
from wilddet3d.vis.visualize import draw_3d_boxes


def cross_category_nms(
    boxes2d, boxes3d, scores, scores_2d, scores_3d, class_ids,
    iou_threshold=0.8,
):
    """Cross-category NMS: suppress overlapping boxes across categories.

    For boxes with 2D IoU > threshold, keep the one with higher combined
    score regardless of category.

    Args:
        boxes2d: (N, 4) tensor, pixel xyxy.
        boxes3d: (N, 10) tensor.
        scores: (N,) combined scores (for ranking).
        scores_2d: (N,) 2D scores.
        scores_3d: (N,) 3D scores.
        class_ids: (N,) class indices.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered tensors (boxes2d, boxes3d, scores, scores_2d, scores_3d,
        class_ids).
    """
    if len(boxes2d) <= 1:
        return boxes2d, boxes3d, scores, scores_2d, scores_3d, class_ids

    # Sort by combined score descending
    order = scores.argsort(descending=True)
    boxes2d = boxes2d[order]
    boxes3d = boxes3d[order]
    scores = scores[order]
    scores_2d = scores_2d[order]
    scores_3d = scores_3d[order]
    class_ids = class_ids[order]

    # Compute pairwise IoU
    x1 = torch.max(boxes2d[:, None, 0], boxes2d[None, :, 0])
    y1 = torch.max(boxes2d[:, None, 1], boxes2d[None, :, 1])
    x2 = torch.min(boxes2d[:, None, 2], boxes2d[None, :, 2])
    y2 = torch.min(boxes2d[:, None, 3], boxes2d[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area = (
        (boxes2d[:, 2] - boxes2d[:, 0]) * (boxes2d[:, 3] - boxes2d[:, 1])
    )
    union = area[:, None] + area[None, :] - inter
    iou = inter / (union + 1e-6)

    n = len(boxes2d)
    suppressed = set()
    keep = []
    for i in range(n):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, n):
            if j in suppressed:
                continue
            if iou[i, j] >= iou_threshold:
                suppressed.add(j)

    keep = torch.tensor(keep, dtype=torch.long, device=boxes2d.device)
    return (
        boxes2d[keep], boxes3d[keep], scores[keep],
        scores_2d[keep], scores_3d[keep], class_ids[keep],
    )


# ---- BEV Renderer (inline JS from bev-renderer.js) ----
BEV_RENDERER_JS = r"""
var BEV_EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
var BEV_FACES=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]];
class BEVRenderer{
constructor(canvasId){this.canvas=canvasId?document.getElementById(canvasId):null;this.ctx=this.canvas?this.canvas.getContext('2d'):null;this.bgColor='#f8f8f8';}
render(boxes,colors,elevDeg){
if(elevDeg===undefined)elevDeg=35;
this._resizeCanvas();var ctx=this.ctx;var w=this.canvas.width;var h=this.canvas.height;
ctx.fillStyle=this.bgColor;ctx.fillRect(0,0,w,h);
var validBoxes=[];var labels=[];var boxColors=[];
for(var i=0;i<boxes.length;i++){var corners=this._getCornersDisplay(boxes[i]);
if(corners){validBoxes.push(corners);labels.push(boxes[i].category||'');boxColors.push(colors[i]||'#e74c3c');}}
if(validBoxes.length===0){ctx.fillStyle='#999';ctx.font='14px Inter,Arial,sans-serif';ctx.textAlign='center';ctx.fillText('No 3D boxes',w/2,h/2);return;}
var allPts=[];for(var i=0;i<validBoxes.length;i++)for(var j=0;j<8;j++)allPts.push(validBoxes[i][j]);
var sceneCenter=this._computeCenter(allPts);var sceneRange=this._computeRange(allPts,sceneCenter);
var distance=sceneRange*1.0;var elev=elevDeg*Math.PI/180;
var eye=[sceneCenter[0],sceneCenter[1]+distance*Math.sin(elev),sceneCenter[2]+distance*Math.cos(elev)];
var viewMat=this._lookAt(eye,sceneCenter);var K=this._computeSmartK(allPts,viewMat,w,h);
var gridElev=35*Math.PI/180;var gridEye=[sceneCenter[0],sceneCenter[1]+distance*Math.sin(gridElev),sceneCenter[2]+distance*Math.cos(gridElev)];
var gridViewMat=this._lookAt(gridEye,sceneCenter);var gridK=this._computeSmartK(allPts,gridViewMat,w,h);
var groundY=this._findGroundY(validBoxes);this._drawGrid(gridViewMat,gridK,w,h,sceneCenter,sceneRange,groundY);
var boxItems=[];for(var i=0;i<validBoxes.length;i++){var center=this._computeCenter(validBoxes[i]);var camPt=this._transformPoint(center,viewMat);boxItems.push({corners:validBoxes[i],depth:-camPt[2],label:labels[i],color:boxColors[i]});}
boxItems.sort(function(a,b){return b.depth-a.depth;});
for(var i=0;i<boxItems.length;i++)this._drawBox3D(boxItems[i].corners,viewMat,K,w,h,boxItems[i].color);
for(var i=0;i<boxItems.length;i++)if(boxItems[i].label)this._drawLabel(boxItems[i].corners,boxItems[i].label,viewMat,K,w,h,boxItems[i].color);
}
_computeSmartK(allPts,viewMat,w,h){var margin=0.10;var camXs=[],camYs=[];
for(var i=0;i<allPts.length;i++){var cam=this._transformPoint(allPts[i],viewMat);var depth=-cam[2];if(depth<=0.01)continue;camXs.push(cam[0]/depth);camYs.push(-cam[1]/depth);}
if(camXs.length===0)return[[w*0.85,0,w/2],[0,w*0.85,h/2],[0,0,1]];
var minNx=Math.min.apply(null,camXs),maxNx=Math.max.apply(null,camXs),minNy=Math.min.apply(null,camYs),maxNy=Math.max.apply(null,camYs);
var rangeNx=maxNx-minNx,rangeNy=maxNy-minNy;if(rangeNx<1e-6)rangeNx=1e-6;if(rangeNy<1e-6)rangeNy=1e-6;
var usableW=w*(1-2*margin),usableH=h*(1-2*margin);var f=Math.min(usableW/rangeNx,usableH/rangeNy);
var midNx=(minNx+maxNx)/2,midNy=(minNy+maxNy)/2;return[[f,0,w/2-f*midNx],[0,f,h/2-f*midNy],[0,0,1]];}
_resizeCanvas(){var c=this.canvas.parentElement;var cw=c.clientWidth,ch=c.clientHeight;if(cw<=0)cw=400;if(ch<=0)ch=400;this.canvas.width=cw;this.canvas.height=ch;}
_getCornersDisplay(box){
var b=box.box3d;if(!b||b.length!==10)return null;
var cx=b[0],cy=b[1],cz=b[2];
var hw=b[3]/2,hl=b[4]/2,hh=b[5]/2;
var qw=b[6],qx=b[7],qy=b[8],qz=b[9];var R=this._quat2rot(qw,qx,qy,qz);
var local=[[-hl,-hh,-hw],[hl,-hh,-hw],[hl,hh,-hw],[-hl,hh,-hw],[-hl,-hh,hw],[hl,-hh,hw],[hl,hh,hw],[-hl,hh,hw]];
var corners=[];for(var i=0;i<8;i++){var lx=local[i][0],ly=local[i][1],lz=local[i][2];
var rx=R[0][0]*lx+R[0][1]*ly+R[0][2]*lz;var ry=R[1][0]*lx+R[1][1]*ly+R[1][2]*lz;var rz=R[2][0]*lx+R[2][1]*ly+R[2][2]*lz;
corners.push([rx+cx,-(ry+cy),-(rz+cz)]);}return corners;}
_quat2rot(qw,qx,qy,qz){return[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]];}
_computeCenter(pts){var sx=0,sy=0,sz=0;for(var i=0;i<pts.length;i++){sx+=pts[i][0];sy+=pts[i][1];sz+=pts[i][2];}var n=pts.length;return[sx/n,sy/n,sz/n];}
_computeRange(pts,center){var m=0;for(var i=0;i<pts.length;i++){var dx=pts[i][0]-center[0],dy=pts[i][1]-center[1],dz=pts[i][2]-center[2];var d=Math.sqrt(dx*dx+dy*dy+dz*dz);if(d>m)m=d;}return Math.max(m*2,2.0);}
_findGroundY(allCorners){var minY=Infinity;for(var i=0;i<allCorners.length;i++)for(var j=0;j<8;j++)if(allCorners[i][j][1]<minY)minY=allCorners[i][j][1];return minY;}
_lookAt(eye,target){var fwd=[target[0]-eye[0],target[1]-eye[1],target[2]-eye[2]];var fLen=Math.sqrt(fwd[0]*fwd[0]+fwd[1]*fwd[1]+fwd[2]*fwd[2]);fwd=[fwd[0]/fLen,fwd[1]/fLen,fwd[2]/fLen];
var up=[0,1,0];var right=this._cross(fwd,up);var rLen=Math.sqrt(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);right=[right[0]/rLen,right[1]/rLen,right[2]/rLen];var trueUp=this._cross(right,fwd);
var m=new Float64Array(16);m[0]=right[0];m[1]=right[1];m[2]=right[2];m[3]=-(right[0]*eye[0]+right[1]*eye[1]+right[2]*eye[2]);
m[4]=trueUp[0];m[5]=trueUp[1];m[6]=trueUp[2];m[7]=-(trueUp[0]*eye[0]+trueUp[1]*eye[1]+trueUp[2]*eye[2]);
m[8]=-fwd[0];m[9]=-fwd[1];m[10]=-fwd[2];m[11]=-(-fwd[0]*eye[0]+-fwd[1]*eye[1]+-fwd[2]*eye[2]);m[12]=0;m[13]=0;m[14]=0;m[15]=1;return m;}
_cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
_transformPoint(pt,mat){return[mat[0]*pt[0]+mat[1]*pt[1]+mat[2]*pt[2]+mat[3],mat[4]*pt[0]+mat[5]*pt[1]+mat[6]*pt[2]+mat[7],mat[8]*pt[0]+mat[9]*pt[1]+mat[10]*pt[2]+mat[11]];}
_project(pt,viewMat,K){var cam=this._transformPoint(pt,viewMat);var depth=-cam[2];if(depth<=0.01)return null;return{x:K[0][0]*cam[0]/depth+K[0][2],y:-K[1][1]*cam[1]/depth+K[1][2],depth:depth};}
_drawGrid(viewMat,K,w,h,center,range,groundY){var ctx=this.ctx;var half=range*0.6;var spacing=Math.max(0.5,range/5);ctx.save();ctx.strokeStyle='#d0d0d0';ctx.lineWidth=1;
var zS=Math.ceil((center[2]-half)/spacing)*spacing;for(var z=zS;z<=center[2]+half;z+=spacing){var p1=this._project([center[0]-half,groundY,z],viewMat,K);var p2=this._project([center[0]+half,groundY,z],viewMat,K);if(p1&&p2){ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();}}
var xS=Math.ceil((center[0]-half)/spacing)*spacing;for(var x=xS;x<=center[0]+half;x+=spacing){var p1=this._project([x,groundY,center[2]-half],viewMat,K);var p2=this._project([x,groundY,center[2]+half],viewMat,K);if(p1&&p2){ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();}}ctx.restore();}
_drawBox3D(corners,viewMat,K,w,h,color){var ctx=this.ctx;var pts2d=[],depths=[];for(var i=0;i<8;i++){var p=this._project(corners[i],viewMat,K);if(!p)return;pts2d.push(p);depths.push(p.depth);}
var faceDepths=[];for(var f=0;f<BEV_FACES.length;f++){var face=BEV_FACES[f];var avg=0;for(var j=0;j<face.length;j++)avg+=depths[face[j]];faceDepths.push({idx:f,d:avg/face.length});}faceDepths.sort(function(a,b){return b.d-a.d;});
ctx.save();ctx.globalAlpha=0.12;ctx.fillStyle=color;for(var fi=0;fi<faceDepths.length;fi++){var face=BEV_FACES[faceDepths[fi].idx];ctx.beginPath();ctx.moveTo(pts2d[face[0]].x,pts2d[face[0]].y);for(var j=1;j<face.length;j++)ctx.lineTo(pts2d[face[j]].x,pts2d[face[j]].y);ctx.closePath();ctx.fill();}ctx.globalAlpha=1.0;
ctx.strokeStyle=color;ctx.lineWidth=2;for(var e=0;e<BEV_EDGES.length;e++){var i0=BEV_EDGES[e][0],i1=BEV_EDGES[e][1];ctx.beginPath();ctx.moveTo(pts2d[i0].x,pts2d[i0].y);ctx.lineTo(pts2d[i1].x,pts2d[i1].y);ctx.stroke();}ctx.restore();}
_drawLabel(corners,label,viewMat,K,w,h,color){var ctx=this.ctx;var minY=Infinity,labelX=0;for(var i=0;i<8;i++){var p=this._project(corners[i],viewMat,K);if(!p)return;if(p.y<minY){minY=p.y;labelX=p.x;}}
var fontSize=11,padH=4,padV=2;ctx.save();ctx.font=fontSize+'px Inter,Arial,sans-serif';var textW=ctx.measureText(label).width;var pillW=textW+padH*2,pillH=fontSize+padV*2;var px=labelX-pillW/2,py=minY-pillH-3;
if(px<2)px=2;if(px+pillW>w-2)px=w-2-pillW;if(py<2)py=2;
ctx.globalAlpha=0.8;ctx.fillStyle=color;var r=3;ctx.beginPath();ctx.moveTo(px+r,py);ctx.lineTo(px+pillW-r,py);ctx.arcTo(px+pillW,py,px+pillW,py+r,r);ctx.lineTo(px+pillW,py+pillH-r);ctx.arcTo(px+pillW,py+pillH,px+pillW-r,py+pillH,r);ctx.lineTo(px+r,py+pillH);ctx.arcTo(px,py+pillH,px,py+pillH-r,r);ctx.lineTo(px,py+r);ctx.arcTo(px,py,px+r,py,r);ctx.closePath();ctx.fill();
ctx.globalAlpha=1.0;ctx.fillStyle='#fff';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(label,px+padH,py+padV);ctx.restore();}
}
"""

# Color palette for BEV boxes (per category)
BEV_COLORS = [
    "#e74c3c", "#3b82f6", "#22c55e", "#f59e0b",
    "#a855f7", "#06b6d4", "#ec4899", "#f97316",
]


def make_bev_html(boxes3d_np, class_ids_np, class_names, elev_deg=35):
    """Generate HTML with embedded BEV canvas renderer.

    Args:
        boxes3d_np: (N, 10) numpy array of 3D boxes.
        class_ids_np: (N,) numpy array of class indices.
        class_names: list of class name strings.
        elev_deg: initial elevation angle in degrees.

    Returns:
        HTML string with embedded canvas + JS.
    """
    boxes_json = []
    colors_json = []
    for i in range(len(boxes3d_np)):
        cid = int(class_ids_np[i])
        name = class_names[cid] if cid < len(class_names) else "object"
        b = boxes3d_np[i]
        label = name
        boxes_json.append({
            "box3d": b.tolist(),
            "category": label,
        })
        colors_json.append(BEV_COLORS[cid % len(BEV_COLORS)])

    boxes_data = json.dumps(boxes_json)
    colors_data = json.dumps(colors_json)

    # Build a self-contained HTML page for the iframe.
    # Gradio gr.HTML uses innerHTML which does NOT execute <script> tags,
    # so we wrap in an <iframe srcdoc="..."> to get script execution.
    inner_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#f8f8f8;font-family:Inter,Arial,sans-serif;overflow:hidden;}}
.controls{{display:flex;align-items:center;gap:12px;padding:6px 12px;
  background:#eee;border-bottom:1px solid #ddd;}}
.controls label{{font-size:13px;color:#555;}}
.controls span{{font-size:13px;font-family:monospace;min-width:30px;}}
.controls input[type=range]{{flex:1;max-width:200px;}}
canvas{{width:100%;height:calc(100vh - 36px);display:block;}}
</style></head><body>
<div class="controls">
  <label>Elevation:</label>
  <input type="range" id="elev" min="-90" max="90" value="{elev_deg}" step="1"
    oninput="document.getElementById('ev').textContent=this.value;draw(+this.value);">
  <span id="ev">{elev_deg}</span>
</div>
<canvas id="bev"></canvas>
<script>
{BEV_RENDERER_JS}
var R=new BEVRenderer('bev');
var B={boxes_data};
var C={colors_data};
function draw(e){{R.render(B,C,e);}}
draw({elev_deg});
new ResizeObserver(function(){{draw(+document.getElementById('elev').value);}}).observe(document.getElementById('bev').parentElement);
</script></body></html>"""

    # Escape for srcdoc attribute (double-quote safe)
    escaped = inner_html.replace("&", "&amp;").replace('"', "&quot;")

    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:450px;border:none;border-radius:8px;" '
        f'sandbox="allow-scripts"></iframe>'
    )


def draw_points_on_image(image, points):
    """Draw points on image.

    Args:
        image: numpy array (H, W, 3)
        points: list of (x, y, label) tuples

    Returns:
        Image with points drawn
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    radius = max(4, int(min(h, w) * 0.012))
    for x, y, label in points:
        # Positive = green, Negative = red
        fill = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(img, (int(x), int(y)), radius, fill, -1)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 2)
    return img


def draw_box_on_image(image, box, thickness=3):
    """Draw box on image.

    Args:
        image: numpy array (H, W, 3)
        box: [x1, y1, x2, y2] coordinates
        thickness: line thickness

    Returns:
        Image with box drawn
    """
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (255, 50, 50)  # bright red in RGB
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


# HuggingFace Model repo for checkpoints
HF_MODEL_REPO = "allenai/WildDet3D"
HF_CKPT_NAME = "wilddet3d_alldata_all_prompt_v1.0.pt"

# Local checkpoint paths (tried in order)
LOCAL_CHECKPOINTS = [
    "ckpt/wilddet3d.pt",  # release repo layout
]

# Default demo image path (relative to this file for local run)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_PATH = os.path.join(_SCRIPT_DIR, "assets/demo/rgb.png")
if not os.path.exists(DEFAULT_IMAGE_PATH):
    DEFAULT_IMAGE_PATH = "assets/demo/rgb.png"  # fallback for HF Space

# Global model (loaded once)
_cached_model = None


def _resolve_checkpoint():
    """Resolve checkpoint: local if exists, else download from HF Hub."""
    for path in LOCAL_CHECKPOINTS:
        if os.path.exists(path):
            return path
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Downloading checkpoint from {HF_MODEL_REPO}...")
    ckpt = hf_hub_download(
        repo_id=HF_MODEL_REPO, filename=HF_CKPT_NAME, token=hf_token
    )
    return ckpt


def get_model():
    """Load model once and cache it."""
    global _cached_model
    if _cached_model is None:
        ckpt_path = _resolve_checkpoint()
        print(f"Loading WildDet3D model from {ckpt_path}...")
        _cached_model = build_model(
            checkpoint=ckpt_path,
            score_threshold=0.0,
            canonical_rotation=True,
            skip_pretrained=True,
        )
        print("Model loaded!")
    return _cached_model


def load_default_image():
    """Load the default demo image."""
    if os.path.exists(DEFAULT_IMAGE_PATH):
        return np.array(Image.open(DEFAULT_IMAGE_PATH).convert("RGB"))
    return None


def load_default_intrinsics():
    """Return placeholder intrinsics values."""
    return 0, 0, 0, 0


def format_intrinsics(K):
    """Format intrinsics tensor for display."""
    if K is None:
        return "Not available"
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    if K.ndim == 3:
        K = K[0]
    return (
        f"fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}, "
        f"cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}"
    )


def scale_intrinsics_to_original(K, input_hw, original_hw):
    """Scale intrinsics from model input resolution to original."""
    if K is None:
        return None

    if isinstance(K, torch.Tensor):
        K = K.clone()
    else:
        K = K.copy()

    input_h, input_w = input_hw
    orig_h, orig_w = original_hw

    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    if K.ndim == 3:
        K[:, 0, 0] *= scale_x
        K[:, 1, 1] *= scale_y
        K[:, 0, 2] *= scale_x
        K[:, 1, 2] *= scale_y
    else:
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

    return K


def on_image_select(
    evt: gr.SelectData, image, original_image, state,
    prompt_mode, point_label,
):
    """Handle click on image and visualize the click."""
    if image is None:
        return state, "Please upload an image first", None

    x, y = evt.index[0], evt.index[1]
    label = 1 if "Positive" in point_label else 0

    new_state = {
        "points": list(state.get("points", [])),
        "box": list(state.get("box", [])),
    }

    vis_image = (
        original_image.copy()
        if original_image is not None
        else image.copy()
    )

    if prompt_mode == "Point":
        new_state["points"].append((x, y, label))
        new_state["box"] = []
        label_str = "+" if label == 1 else "-"
        info = (
            f"Points: {len(new_state['points'])} total. "
            f"Last: ({x}, {y}) [{label_str}]"
        )
        vis_image = draw_points_on_image(vis_image, new_state["points"])

    elif prompt_mode in ("Box-to-Multi-Object", "Box-to-Single-Object"):
        new_state["points"] = []
        box_clicks = list(new_state.get("box", []))
        box_clicks.append((x, y))

        if len(box_clicks) == 1:
            new_state["box"] = box_clicks
            info = (
                f"[{prompt_mode}] Corner 1: ({x}, {y}) "
                f"- click again for corner 2"
            )
            vis_image = draw_points_on_image(vis_image, [(x, y, 1)])

        elif len(box_clicks) >= 2:
            x1, y1 = box_clicks[0]
            x2, y2 = box_clicks[1]
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            new_state["box"] = [(box[0], box[1]), (box[2], box[3])]
            info = (
                f"[{prompt_mode}] Box: "
                f"({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})"
            )
            vis_image = draw_box_on_image(vis_image, box)
        else:
            info = f"Box clicks: {box_clicks}"
    else:
        info = "Text mode - just enter text and click Run"

    return new_state, info, vis_image


def clear_clicks(state, original_image):
    """Reset click state and restore original image."""
    new_state = {"points": [], "box": []}
    return (
        new_state,
        "Cleared - ready for new clicks",
        original_image.copy() if original_image is not None else None,
    )


@spaces.GPU
def run_wilddet3d(
    image,
    original_image,
    state,
    prompt_mode,
    text_prompt,
    use_label,
    label_text,
    score_thres,
    use_actual_K,
    fx, fy, cx, cy,
):
    """Run WildDet3D with selected prompt mode."""
    if image is None:
        return None, "Please upload an image first", None, ""

    # Use original (clean) image for model input in point/box modes,
    # since `image` may have drawn points/boxes on it.
    if original_image is not None and prompt_mode != "Text":
        image = original_image

    # Convert RGBA to RGB if needed
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = get_model()

    # Build intrinsics matrix
    if use_actual_K:
        if fx <= 0 or fy <= 0:
            return (
                None,
                "Please enter valid intrinsics (fx, fy must be > 0)",
                None,
                None,
            )
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        intrinsics = None  # preprocess uses default placeholder

    # Preprocess image
    data = preprocess(image.astype(np.float32), intrinsics)

    # Build prompt_text for box/point modes
    if prompt_mode == "Box-to-Multi-Object":
        prefix = "visual"
    elif prompt_mode == "Box-to-Single-Object":
        prefix = "geometric"
    else:
        prefix = "geometric"  # Point mode default

    if prompt_mode != "Text":
        if use_label and label_text and label_text.strip():
            geo_prompt_text = f"{prefix}: {label_text.strip()}"
        else:
            geo_prompt_text = prefix

    # Initialize prompt info for visualization
    prompt_points = None
    prompt_box = None

    # Run based on prompt mode
    if prompt_mode == "Text":
        input_texts = [
            t.strip() for t in text_prompt.split(".") if t.strip()
        ]
        if not input_texts:
            input_texts = ["object"]

        results = detector(
            images=data["images"].to(device),
            intrinsics=data["intrinsics"].to(device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_texts=input_texts,
            return_predicted_intrinsics=True,
        )
        (
            boxes, boxes3d, scores, scores_2d, scores_3d,
            class_ids, depth_maps, predicted_K, confidence_maps,
        ) = results
        class_id_mapping = {i: t for i, t in enumerate(input_texts)}

    elif prompt_mode in ("Box-to-Multi-Object", "Box-to-Single-Object"):
        box_coords = state.get("box", [])
        if len(box_coords) < 2:
            return (
                None,
                "Please click twice on the image to define a box",
                None,
                None,
            )

        x1_orig, y1_orig = box_coords[0]
        x2_orig, y2_orig = box_coords[1]
        box_xyxy = [
            float(x1_orig), float(y1_orig),
            float(x2_orig), float(y2_orig),
        ]

        prompt_box = [x1_orig, y1_orig, x2_orig, y2_orig]

        results = detector(
            images=data["images"].to(device),
            intrinsics=data["intrinsics"].to(device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_boxes=[box_xyxy],
            prompt_text=geo_prompt_text,
            return_predicted_intrinsics=True,
        )
        (
            boxes, boxes3d, scores, scores_2d, scores_3d,
            class_ids, depth_maps, predicted_K, confidence_maps,
        ) = results
        class_id_mapping = {0: geo_prompt_text}

    elif prompt_mode == "Point":
        points = state.get("points", [])
        if not points:
            return (
                None,
                "Please click on the image to select a point",
                None,
                None,
            )

        prompt_points = points

        results = detector(
            images=data["images"].to(device),
            intrinsics=data["intrinsics"].to(device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_points=[points],
            prompt_text=geo_prompt_text,
            return_predicted_intrinsics=True,
        )
        (
            boxes, boxes3d, scores, scores_2d, scores_3d,
            class_ids, depth_maps, predicted_K, confidence_maps,
        ) = results
        class_id_mapping = {0: geo_prompt_text}

    else:
        return None, f"Unknown prompt mode: {prompt_mode}", None, None

    # Cross-category NMS (suppress duplicates across categories)
    if len(boxes[0]) > 1:
        (
            boxes[0], boxes3d[0], scores[0],
            scores_2d[0], scores_3d[0], class_ids[0],
        ) = cross_category_nms(
            boxes[0], boxes3d[0], scores[0],
            scores_2d[0], scores_3d[0], class_ids[0],
            iou_threshold=0.8,
        )

    # For one-to-one modes (Point, Box-to-Single-Object), keep only
    # the highest confidence detection.
    if prompt_mode in ("Point", "Box-to-Single-Object") and len(boxes[0]) > 1:
        best = scores[0].argmax()
        boxes[0] = boxes[0][best:best+1]
        boxes3d[0] = boxes3d[0][best:best+1]
        scores[0] = scores[0][best:best+1]
        scores_2d[0] = scores_2d[0][best:best+1]
        scores_3d[0] = scores_3d[0][best:best+1]
        class_ids[0] = class_ids[0][best:best+1]

    # Scale predicted intrinsics to original resolution
    predicted_K_scaled = scale_intrinsics_to_original(
        predicted_K,
        input_hw=data["input_hw"],
        original_hw=data["original_hw"],
    )

    # Format intrinsics info
    orig_h, orig_w = data["original_hw"]
    intrinsics_info = f"Image: {orig_w}x{orig_h}\n"
    if use_actual_K:
        intrinsics_info += (
            f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, "
            f"cx={cx:.2f}, cy={cy:.2f}"
        )
    else:
        intrinsics_info += (
            f"Intrinsics: default (focal={max(orig_h, orig_w)})"
        )

    # 2D visualization
    img_2d = visualize_results(
        data, boxes3d, scores, scores_2d, scores_3d,
        class_ids, class_id_mapping, score_thres,
    )

    # Depth map visualization (with confidence mask if available)
    depth_vis_img = None
    if depth_maps is not None and len(depth_maps) > 0:
        depth_np_raw = depth_maps[0].cpu().numpy()
        d = depth_np_raw.squeeze()

        pad_l, pad_r, pad_t, pad_b = data["padding"]
        h_end = d.shape[0] - pad_b if pad_b > 0 else d.shape[0]
        w_end = d.shape[1] - pad_r if pad_r > 0 else d.shape[1]
        d_crop = d[pad_t:h_end, pad_l:w_end]

        # Get confidence mask if available
        conf_mask = None
        if confidence_maps is not None and len(confidence_maps) > 0:
            c = confidence_maps[0].cpu().numpy().squeeze()
            c_crop = c[pad_t:h_end, pad_l:w_end]
            conf_mask = c_crop > 0.5

        d_valid = d_crop[d_crop > 0.01]
        if len(d_valid) > 0:
            d_min, d_max = d_valid.min(), d_valid.max()
            d_norm = np.clip(
                (d_crop - d_min) / (d_max - d_min + 1e-6), 0, 1
            )
            d_norm = (1.0 - d_norm) * 255
            d_norm = d_norm.astype(np.uint8)
            depth_vis_img = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            depth_vis_img = cv2.cvtColor(depth_vis_img, cv2.COLOR_BGR2RGB)

            # Apply confidence mask: low-confidence regions -> gray
            if conf_mask is not None:
                gray_bg = np.full_like(depth_vis_img, 200)
                depth_vis_img = np.where(
                    conf_mask[:, :, None], depth_vis_img, gray_bg
                )

            depth_vis_img = Image.fromarray(depth_vis_img)

    # BEV visualization
    bev_html = ""
    mask_bev = scores_2d[0] >= score_thres
    if mask_bev.sum() > 0:
        bev_boxes = boxes3d[0][mask_bev].cpu().numpy()
        bev_cids = class_ids[0][mask_bev].cpu().numpy()
        bev_names = [
            class_id_mapping.get(i, str(i))
            for i in range(max(len(class_id_mapping), 1))
        ]
        bev_html = make_bev_html(bev_boxes, bev_cids, bev_names)

    return img_2d, intrinsics_info, depth_vis_img, bev_html


def visualize_results(
    data, boxes3d, scores, scores_2d, scores_3d, class_ids,
    class_id_mapping, score_thres,
):
    """Visualize 3D detection results using wilddet3d.vis.draw_3d_boxes."""
    filtered_boxes3d = []
    filtered_scores_2d = []
    filtered_scores_3d = []
    filtered_class_ids = []

    for i in range(len(boxes3d)):
        mask = scores_2d[i] >= score_thres
        filtered_boxes3d.append(boxes3d[i][mask])
        if scores_2d is not None:
            filtered_scores_2d.append(scores_2d[i][mask])
        else:
            filtered_scores_2d.append(torch.zeros_like(scores[i][mask]))
        if scores_3d is not None:
            filtered_scores_3d.append(scores_3d[i][mask])
        else:
            filtered_scores_3d.append(torch.zeros_like(scores[i][mask]))
        filtered_class_ids.append(class_ids[i][mask])

    # Get original image (clean, no prompt annotations)
    # original_images is (1, 3, H, W) float32 [0, 255] after ToTensor
    # -> squeeze batch, permute to (H, W, 3) for visualization
    original_img = (
        data["original_images"]
        .cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    )

    # Use wilddet3d's draw_3d_boxes for visualization
    K = data["original_intrinsics"].cpu().numpy()
    if K.ndim == 3:
        K = K[0]

    class_names = [
        class_id_mapping.get(i, str(i))
        for i in range(max(len(class_id_mapping), 1))
    ]

    # Draw 3D boxes with 2D/3D score labels
    if len(filtered_boxes3d) > 0 and len(filtered_boxes3d[0]) > 0:
        pil_img = draw_3d_boxes(
            image=original_img,
            boxes3d=filtered_boxes3d[0],
            intrinsics=K,
            scores_2d=filtered_scores_2d[0],
            scores_3d=filtered_scores_3d[0],
            class_ids=filtered_class_ids[0],
            class_names=class_names,
            n_colors=max(len(class_id_mapping), 1),
        )
    else:
        pil_img = Image.fromarray(original_img)

    return pil_img


# Load default values
default_fx, default_fy, default_cx, default_cy = load_default_intrinsics()
default_image = load_default_image()

# Build Gradio interface
with gr.Blocks(
    title="WildDet3D: 3D Detection",
    css="""
        .column-form { border: none !important; }
        .gradio-container { max-width: 100% !important; padding: 0 !important; }
        .contain { max-width: 100% !important; }
        * { outline: none !important; }
        .gr-group, .gr-box, .gr-panel,
        [class*="column"], [class*="Column"] {
            border-color: transparent !important;
            box-shadow: none !important;
        }
    """,
) as demo:

    # ---- Terms of Use gate ----
    with gr.Column(visible=True) as terms_page:
        gr.Markdown(
            "# WildDet3D: Scaling Promptable 3D Detection in the Wild\n"
            "### Allen Institute for AI (Ai2)"
        )
        gr.Markdown("""
### WildDet3D Terms of Use

By using WildDet3D, you agree:

- to <a href="https://allenai.org/terms" target="_blank" rel="noopener">Ai2's Terms of Use</a> and <a href="https://allenai.org/responsible-use" target="_blank" rel="noopener">Responsible Use Guidelines</a>;
- you will not submit or upload personal, sensitive, confidential, or proprietary information to WildDet3D; and
- none of your uploaded content or inputs to WildDet3D will violate <a href="https://huggingface.co/code-of-conduct" target="_blank" rel="noopener">Hugging Face's Code of Conduct</a> or <a href="https://huggingface.co/content-policy" target="_blank" rel="noopener">Content Policy</a>.

**If you do not agree with any of these statements, please do not access or use WildDet3D.**
""")
        agree_btn = gr.Button(
            "Agree & Use WildDet3D",
            variant="primary",
            size="lg",
        )

    # ---- Main app (hidden until agreed) ----
    with gr.Column(visible=False) as main_app:
        gr.Markdown(
            "# WildDet3D: Scaling Promptable 3D Detection in the Wild\n"
            "### Allen Institute for AI (Ai2)"
        )
        gr.Markdown("""
**How to use:**
- **Text**: Enter object names (e.g., "car.person.traffic light"), click Run
- **Box-to-Multi-Object**: Draw box -> detect ALL similar objects (one-to-many)
- **Box-to-Single-Object**: Draw box -> detect ONLY the boxed object (one-to-one)
- **Point**: Click on object, click Run
""")

        # State for click coordinates and original image
        click_state = gr.State({"points": [], "box": []})
        original_image_state = gr.State(
            default_image.copy() if default_image is not None else None
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image (click for Box/Point mode)",
                    type="numpy",
                    value=default_image,
                    interactive=True,
                    sources=["upload", "clipboard"],
                )

                # Prompt settings
                prompt_mode = gr.Radio(
                    choices=[
                        "Text",
                        "Box-to-Multi-Object",
                        "Box-to-Single-Object",
                        "Point",
                    ],
                    value="Box-to-Single-Object",
                    label="Prompt Mode",
                )
                text_prompt = gr.Textbox(
                    label="Text Prompt (separate categories with '.')",
                    value="person.chair.monitor.pen",
                    placeholder="e.g. chair.table.monitor",
                    visible=False,
                )
                # Box mode caption
                box_caption = gr.Markdown(
                    "Click the top-left corner, then the bottom-right corner to draw a box.",
                    visible=True,
                )
                # Point mode caption
                point_caption = gr.Markdown(
                    "Click on the image to add points. Use positive to include, negative to exclude.",
                    visible=False,
                )
                # Point mode controls
                point_label = gr.Radio(
                    choices=["Positive (include)", "Negative (exclude)"],
                    value="Positive (include)",
                    label="Point Label",
                    visible=False,
                )
                # Hidden states (kept for function signatures)
                use_label = gr.Checkbox(value=False, visible=False)
                label_text = gr.Textbox(value="", visible=False)
                click_info = gr.Textbox(value="", visible=False)

                with gr.Row():
                    clear_btn = gr.Button(
                        "Clear Clicks", visible=True
                    )
                    run_btn = gr.Button("Run Detection", variant="primary")

                # Intrinsics settings
                use_actual_K = gr.Checkbox(
                    label="Use Actual Intrinsics (uncheck to use default)",
                    value=False,
                )
                with gr.Row(visible=False) as intrinsics_row:
                    fx = gr.Number(label="fx", value=0)
                    fy = gr.Number(label="fy", value=0)
                    cx = gr.Number(label="cx", value=0)
                    cy = gr.Number(label="cy", value=0)

                score_thres = gr.Slider(
                    minimum=0, maximum=1, value=0.3, step=0.05,
                    label="Score Threshold",
                )

            # Right column: Output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="3D Detection Results", type="pil"
                )
                bev_output = gr.HTML(
                    show_label=False,
                    value="<div style='height:450px;background:#f8f8f8;"
                    "display:flex;align-items:center;justify-content:center;"
                    "font-family:Inter,Arial,sans-serif;color:#999;"
                    "border-radius:8px;'>"
                    "BEV view will appear after detection</div>",
                )
                gr.Markdown(
                    "*Predictions filtered by per-category NMS and "
                    "cross-category NMS. "
                    "For object shape and location, this is raw model "
                    "output without alignment with point cloud "
                    "or the model's own predicted depth.*",
                )
                depth_image = gr.Image(label="Model Predicted Depth Map", type="pil")
                intrinsics_info = gr.Textbox(
                    label="Intrinsics Info", interactive=False
                )

        # Toggle visibility based on prompt mode
        def on_mode_change(mode, orig_img):
            is_text = mode == "Text"
            is_point = mode == "Point"
            is_box = mode in ("Box-to-Multi-Object", "Box-to-Single-Object")
            new_state = {"points": [], "box": []}
            restored_img = orig_img.copy() if orig_img is not None else None
            return (
                gr.update(visible=is_text),
                gr.update(visible=is_box),
                gr.update(visible=is_point),
                gr.update(visible=is_point),
                gr.update(visible=not is_text),
                new_state,
                restored_img,
            )

        prompt_mode.change(
            on_mode_change,
            inputs=[prompt_mode, original_image_state],
            outputs=[
                text_prompt, box_caption, point_caption, point_label,
                clear_btn, click_state, input_image,
            ],
        )

        # Toggle intrinsics input visibility
        def on_intrinsics_toggle(use_actual):
            return gr.update(visible=use_actual)

        use_actual_K.change(
            on_intrinsics_toggle,
            inputs=[use_actual_K],
            outputs=[intrinsics_row],
        )

        # Connect events
        input_image.select(
            on_image_select,
            inputs=[
                input_image, original_image_state, click_state,
                prompt_mode, point_label,
            ],
            outputs=[click_state, click_info, input_image],
        )

        clear_btn.click(
            clear_clicks,
            inputs=[click_state, original_image_state],
            outputs=[click_state, click_info, input_image],
        )

        # When new image is uploaded, save it as original
        def on_image_upload(image):
            if image is None:
                return None, {"points": [], "box": []}, "Upload an image"
            return (
                image.copy(),
                {"points": [], "box": []},
                "Image loaded - select mode and click",
            )

        input_image.upload(
            on_image_upload,
            inputs=[input_image],
            outputs=[original_image_state, click_state, click_info],
        )

        run_btn.click(
            run_wilddet3d,
            inputs=[
                input_image, original_image_state, click_state,
                prompt_mode, text_prompt,
                use_label, label_text, score_thres, use_actual_K,
                fx, fy, cx, cy,
            ],
            outputs=[output_image, intrinsics_info, depth_image, bev_output],
        )

    # ---- Terms agree handler ----
    def on_agree():
        return gr.update(visible=False), gr.update(visible=True)

    agree_btn.click(
        on_agree,
        inputs=[],
        outputs=[terms_page, main_app],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WildDet3D Web Demo")
    print("=" * 60)
    print()
    print("Starting server...")
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)
