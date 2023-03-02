from flask import Blueprint, redirect, render_template, request, session
from threading import Thread
from website import app, auth, db
import sys


sys.path.append("C:/Users/Aly khairy/Desktop/Lo-Go_Logo-on-the-go/web app/website/generator")
views = Blueprint('views', __name__)

@views.route('/', methods=['POST', 'GET'])
@views.route('/index', methods=['POST', 'GET'])
@views.route('/home', methods=['POST', 'GET'])
def index():
    return render_template("index.html")


@views.route('/logo-details', methods=['POST', 'GET'])
def logo_details():
    return render_template("logo-details.html")


@views.route('/input-name', methods=['POST', 'GET'])
def input_name():
    return render_template("input/name.html")


@views.route('/input-age', methods=['POST', 'GET'])
def input_age():
    session['class'] = request.form['class']
    return render_template("input/age.html")


@views.route('/input-class', methods=['POST', 'GET'])
def input_class():
    session['gender'] = request.form['gender']
    return render_template("input/class.html")


@views.route('/input-gender', methods=['POST', 'GET'])
def input_gender():
    session['domain'] = request.form['domain']
    return render_template("input/gender.html", )


@views.route('/input-domain', methods=['POST', 'GET'])

def input_domain():
    print(request.form['name'])
    session['name'] = request.form['name']
    session['slogan'] = request.form.get('slogan')
    return render_template("input/domain.html")


@views.route('/input-color', methods=['POST', 'GET'])
def input_color():
    session['style'] = request.form['style']
    return render_template("input/color.html")


@views.route('/input-style', methods=['POST', 'GET'])
def input_style():
    session['age'] = request.form['age']

    return render_template("input/style.html")


@views.route('/output', methods=['POST', 'GET'])
def output():
    return render_template("output.html")


@views.route('/generating-logo', methods=['POST', 'GET'])
def generating_logo():
    session['color'] = request.form['color']

    from generate import generate_images

    t = Thread(target=generate_images)
    t.start()

    return render_template("loading.html")

@views.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        try:
            user = auth.sign_in_with_email_and_password(request.form['email'],request.form['password'])
            
            return redirect("/")
        except:
            print("could not login")

    return render_template("auth/login-page.html")

@views.route('/sign-up', methods=['POST', 'GET'])
def sign_up():
    if request.method == 'POST':
        try:
            user = auth.create_user_with_email_and_password(request.form['email'],request.form['password'])
            db.child("user").push
            return redirect("/")
        except:
            print("could not sign up")


    return render_template("auth/signup-page.html")

@views.route('/logout', methods=['POST', 'GET'])
def logout():
    return "logout"




gender_list = ['male', 'female', 'both']
social_class_list = ['lower', 'middle', 'upper', 'all']
age_list = ['child', 'teen', 'young adult', 'adult', 'elder', 'all']
domain_list = ['agriculture', 'food and natural resources', 'architecture and construction', 'arts', 'audio and video technology',
               'communication', 'business and finance', 'education and training', 'government and public administration', 'medicine',
               'software engineering', 'law', 'public safety and security', 'marketing', 'science', 'technology', 'engineering and math', 'animals', 'transportation', 'accessories and fashion', 'games and entertainment']
color_list = ['black', 'grey', 'blue', 'purple', 'red',
              'brown', 'green', 'pink', 'orange', 'yellow']


def encode_labels(type):
    if (type == "icon"):
        gender = gender_list.index(session['gender'])
        social_class = social_class_list.index(session['class'])
        age = age_list.index(session['age'])
        domain = domain_list.index(session['domain'])
        color = color_list.index(session['color'])
        user_label = f"{gender},{social_class},{age},{domain},{color}" 
        return user_label
    if (type == "text"):
        user_label = [
            session['age'], session['gender'], session['class'], session['domain'], session['name'], session['slogan'],session['style'],session['color']]
        return user_label
    else:
        print(session)
        print("ERROR LOADING SESSION VARIABLES")
