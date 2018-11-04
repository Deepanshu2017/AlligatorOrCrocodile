from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    imagenet_stats,
    get_transforms,
    models,
)
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

# You path where you have stored models/weights.pth
path = Path("data")

# Classes
classes = ["alligator", "crocodile"]

# Create a DataBunch
data = ImageDataBunch.single_from_classes(
    path,
    classes,
    tfms=get_transforms(),
    size=224,
).normalize(imagenet_stats)

# Create a learner and load the weights
learn = create_cnn(data, models.resnet34)
learn.load("stage-2")


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    """
        Predict function which will be called by either classify_url or by upload
        and return the True class as well as the scores. 
        Note: scores are not probabilites, and we may use activation like Softmax
              or sigmoid to convert these scores into probabilities.
    """
    img = open_image(BytesIO(bytes))
    class_,predictions, losses = learn.predict(img)
    return JSONResponse({
        "class": class_,
        "scores": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h3>This app will classify Alligator vs Crocodial<h3>
        <h4>Note: I created this app with total on 400 images so it you may need to provide good quality images showing face of animal"<h4>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    # To run this app start application on server with python
    # python FILENAME serve
    # ex: python server.py server
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8082)
