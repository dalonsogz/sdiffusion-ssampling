from flask import Flask, render_template, request, session
from PIL import Image
import secrets
from diffusers import StableDiffusionPipeline
import torch
import cv2
from datetime import datetime
from jinja2 import Environment


def environment(**options):
    env = Environment(**options, extensions=['jinja2.ext.loopcontrols'])
    return env


print(torch.cuda.is_available())

app = Flask(__name__)

# generate random secret key
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')

pipeline = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-4",
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipeline.enable_freeu(b1=1.3, b2=1.4, s1=0.9, s2=0.2)

super_res = cv2.dnn_superres.DnnSuperResImpl.create()
super_res.readModel("EDSR_x4.pb")
super_res.setModel("edsr", 4)

num_results = 4


@app.route('/')
def hello():
    # home page

    if "current_prompt_input" not in session:
        session["current_prompt_input"] = ""

    if "current_prompt_timestamp" not in session:
        session["current_prompt_timestamp"] = ""

    return render_template(
        "index.html",
        # pass variables into the HTML template
        btn_range=range(num_results),
        prompt_images=["/static/images/placeholder_image.png" for _ in range(num_results)],
        current_prompt_input=session["current_prompt_input"]
    )


@app.route('/prompt', methods=['POST', 'GET'])
def prompt():
    from flask import session
    session["current_timestamp"] = str(datetime.today()).replace(".", "").replace(":", "").replace("-", "").replace(" ", "")
    session["current_prompt_input"] = request.form['prompt_input']

    # generate images from user prompt
    print("user prompt received:", session["current_prompt_input"])

    for i in range(num_results):
        image = pipeline(session["current_prompt_input"]).images[0]
        image.save("static/images/temp/img_" + session["current_timestamp"] + "_" + str(i) + ".png")
        print("saved 'static/images/temp/img_" + session["current_timestamp"] + "_" + str(i) + ".png'")

    return render_template(
        "index.html",
        # pass variables into the HTML template
        btn_range=range(num_results),
        prompt_images=["static/images/temp/img_" + session["current_timestamp"] + "_" + str(i) + ".png" for i in range(num_results)],
        current_prompt_input=session["current_prompt_input"]
    )


@app.route('/supersample', methods=['POST', 'GET'])
def supersample():
    # enlarge and save prompt image in high quality
    print("save button", request.form['save_btn'], "was clicked!")

    temp_img = cv2.imread("./static/images/temp/img_" + session["current_timestamp"] + "_" + str(request.form['save_btn']) + ".png")
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    XL_img = super_res.upsample(temp_img)
    XL_img = Image.fromarray(XL_img)
    XL_img.save("./static/images/saved/img_" + session["current_timestamp"] + "_" + request.form['save_btn'] + ".png")

    print("saved 'static/images/saved/img_" + session["current_timestamp"] + "_" + request.form['save_btn'] + ".png'")

    return render_template(
        "index.html",
        # pass variables into the HTML template
        btn_range=range(num_results),
        prompt_images=["static/images/temp/img_" + session["current_timestamp"] + "_" + str(i) + ".png" for i in range(num_results)],
        current_prompt_input=session["current_prompt_input"]
    )


if __name__ == '__main__':
    # run application
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )
