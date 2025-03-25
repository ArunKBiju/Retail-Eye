from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
import random
from .models import *
import numpy as np
import cv2
import os
from PIL import Image
from datetime import datetime, timedelta
from django.contrib.auth.decorators import login_required
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create your views here.


def home(request):
    return render(request, "home.html")


def registration(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("conf_password")
        print(name, email, password, confirm_password)
        if password == confirm_password:

            if User.objects.filter(username=email).exists():
                messages.warning(request, "Email exist")

            else:
                def generate_customer_id():
                    while True:
                        customer_id = random.randint(100000000, 999999999)
                        if not Customer.objects.filter(customer_id=customer_id).exists():
                            return customer_id
                customer_id = generate_customer_id()
                user = User.objects.create_user(first_name=name, username=email, password=password)
                user.save()
                customer = Customer.objects.create(user=user, customer_id=customer_id)
                customer.save()
                request.session['customer_id'] = customer_id
                print("Data Inserted")

                return redirect(face_image)

        else:
            messages.warning(request, "Password mismatch")

    return render(request, "registration.html")


def face_image(request):
    customer_id = request.session.get('customer_id')
    print(customer_id, "hello")
    if request.method == "POST":
        if not os.path.exists(r'D:\SST\Real Shop\RealShop\media'):
            os.makedirs(r'D:\SST\Real Shop\RealShop\media')

        faceCascade = cv2.CascadeClassifier(
            r'D:\SST\Real Shop\RealShop\haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
        count = 0

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # Save the captured image into the images directory
                cv2.imwrite("D:/SST/Real Shop/RealShop/media/" + str(customer_id) + '.' + str(
                    count) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('image', img)
            # Press Escape to end the program.
            k = cv2.waitKey(100) & 0xff
            if k < 100:
                break
            # Take 100 face samples and stop video. You may increase or decrease the number of images.
            # More images better while training the model.
            elif count >= 100:
                break

        print("\n [INFO] Exiting Program.")
        cam.release()
        cv2.destroyAllWindows()

        path = "D:/SST/Real Shop/RealShop/media/"
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Haar cascade file

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                #         print(imagePath)
                # convert it to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[0])
                faces = faceCascade.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print("\n[INFO] Training faces...")
        faces, ids = getImagesAndLabels(path)
        print(ids)
        recognizer.train(faces, np.array(ids))
        # Save the model into the current directory.
        recognizer.write('D:/SST/Real Shop/RealShop/trainer.yml')

        return redirect(user_login)
    return render(request, "face_image.html", {'customer_id': customer_id})


def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        print(email, password)
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            print(f"User {email} logged in successfully.")

            # Store aspirant ID in session for use in face_identity
            request.session['customer_id'] = user.id
            print("Email and password validated. Proceed to face recognition.")
            return redirect(face_identity)

        else:
            messages.warning(request, "Invalid email or password")
    return render(request, "login.html")


def face_identity(request):
    customer_id = request.session.get('customer_id')
    print(customer_id, "hello")
    if not customer_id:
        messages.warning(request, "Session expired. Please log in again.")
        return redirect(user_login)

    # Fetch the customer based on session ID
    customer = Customer.objects.filter(user_id=customer_id).first()
    if not customer:
        messages.warning(request, "No customer found. Please register.")
        return redirect(registration)

    label = ''
    # deep learning code starts
    detector = cv2.CascadeClassifier(
        r'D:\SST\Real Shop\RealShop\haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("D:/SST/Real Shop/RealShop/trainer.yml")

    # Load the image
    cam = cv2.VideoCapture(0)
    face_recognized = False  # Track face recognition status

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            cv2.putText(img, "No face detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for (x, y, w, h) in faces:
            # Predict the ID of the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            print(f"Detected ID: {id}, Confidence: {confidence}")
            # Check if confidence is less than 100 ==> "0" is perfect match
            if id == customer.customer_id and confidence < 80:
                label = f"Face Verified: Reg ID {id}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Face Recognition", img)
                cam.release()
                cv2.destroyAllWindows()
                return redirect(dashboard, customer_id=customer.id)
            else:
                label = "Unrecognized Face!"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return redirect(registration)
        # Display the frame
        cv2.imshow('Face Recognition', img)

        # Allow quitting the recognition process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup and redirect if face is not recognized
    cam.release()
    cv2.destroyAllWindows()


def dashboard(request, customer_id):
    # Fetch the customer using customer_id
    customer = Customer.objects.filter(id=customer_id).first()

    if not customer:
        messages.error(request, "Customer not found.")
        return redirect('user_login')  # Ensure 'user_login' is a valid URL name

    # Get or create a wallet for the specific customer
    wallet, created = Wallet.objects.get_or_create(customer=customer)

    return render(request, "dashboard.html", {
        'is_authenticated': True,
        "customer_name": customer.user.first_name,
        "customer_id": customer.customer_id,
        'wallet': wallet
    })


def otp_generate(request):
    if request.method == "POST":
        email = request.POST.get("email")
        print(email)
        if User.objects.filter(username=email).exists():
            def generate_otp():
                return random.randint(1000, 9999)

            otp = generate_otp()
            send_mail("OTP for Password Reset", f"Your OTP for forgot password verification is {otp}",
                      settings.EMAIL_HOST_USER, [email])
            request.session['otp'] = otp
            request.session['email'] = email
            request.session['time'] = str(datetime.now())
            return redirect(password_reset)
        else:
            messages.warning(request, "Account not Found")
    return render(request, "otp_generate.html")


def password_reset(request):
    otp = request.session.get('otp')
    print(otp)
    email = request.session.get('email')
    send_time = request.session.get('time')
    send_time = datetime.strptime(send_time, "%Y-%m-%d %H:%M:%S.%f")
    current_time = datetime.now()
    duration = current_time - send_time
    print(duration)
    if request.method == "POST":
        otp1 = int(request.POST.get("otp"))
        print(otp1)
        new_password = request.POST.get("password")
        re_password = request.POST.get("conf_password")

        if new_password == re_password:
            if otp == otp1 and duration <= timedelta(minutes=5):
                if otp == otp1 and duration <= timedelta(minutes=5):
                    user = User.objects.get(username=email)
                    user.set_password(new_password)
                    user.save()
                    return redirect(user_login)
                elif otp == otp1 and duration > timedelta(minutes=5):
                    messages.warning(request, "Time exceeded !!")
                else:
                    messages.warning(request, "Invalid otp")
            else:
                messages.warning(request, "Password Mismatch")
    return render(request, "password_reset.html")


def user_logout(request):
    logout(request)
    return redirect(home)


def product_scan(request):
    if request.method == "POST":

        product_prices = {
            'milk': 26,
            'toothbrush': 30,
            'apple': 25,
            'banana': 12,
            'bread': 45,
            'carrot': 8,
            'cup': 50,
            'orange': 15,
            'shoe': 250,
            'spoon': 85
        }

        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=r"D:\SST\Real Shop\RealShop\best1.pt",
                               force_reload=True, device='cpu')

        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        total_price = 0
        detected_products = set()

        Items.objects.all().delete()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame_rgb)

            current_frame_products = set()
            for *box, conf, cls in results.pred[0]:
                product_name = model.names[int(cls)]

                if product_name in product_prices and conf > 0.60:
                    current_frame_products.add(product_name)

            new_detections = current_frame_products - detected_products

            for product_name in new_detections:
                price = product_prices[product_name]
                total_price += price
                detected_products.add(product_name)
                print(f"Detected {product_name}: {price}")
                shop_item = Items.objects.create(item=product_name, price=price)
                shop_item.save()
                print(f"Total price so far: {total_price}")

            results.render()

            frame_with_boxes = results.ims[0]

            frame_with_boxes_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

            cv2.imshow('Object Detection', frame_with_boxes_bgr)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Print final total price when the camera stops
        print(f"Final total price: {total_price}")
        request.session["total_price"] = total_price
        return redirect(billing_session)
    return render(request, "product_scan.html")


def view_bill(request):
    total_price = request.session.get("total_price")
    view_bills = Items.objects.all()
    return render(request, "view_bill.html", {"view_bills": view_bills, "total_price": total_price})


def return_product(request):
    if request.method == "POST":
        product_prices = {
            'milk': 26,
            'toothbrush': 30,
            'apple': 25,
            'banana': 12,
            'bread': 45,
            'carrot': 8,
            'cup': 50,
            'orange': 15,
            'shoe': 250,
            'spoon': 85
        }

        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=r"D:\SST\Real Shop\RealShop\best1.pt",
                               force_reload=True, device='cpu')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        total_price = request.session.get("total_price", 0)
        detected_products = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            current_frame_products = set()

            for *box, conf, cls in results.pred[0]:
                product_name = model.names[int(cls)]
                if product_name in product_prices and conf > 0.60:
                    current_frame_products.add(product_name)

            return_detections = current_frame_products - detected_products
            for product_name in return_detections:
                # Check if product exists in the Items table
                item = Items.objects.filter(item=product_name).first()
                if item:
                    total_price -= item.price
                    detected_products.add(product_name)
                    print(f"Returned {product_name}: -{item.price}")

                    # Remove the item from the database
                    item.delete()
                    print(f"Total price after return: {total_price}")
                else:
                    print(f"{product_name} not found in the bill.")

            results.render()
            frame_with_boxes = results.ims[0]
            frame_with_boxes_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imshow('Return Product', frame_with_boxes_bgr)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Update the total price in the session
        request.session["total_price"] = total_price
        print(f"Final total price after return: {total_price}")

    return render(request, "return_product.html")


@login_required
def wallet_view(request):
    customer = Customer.objects.get(user=request.user)  # Get logged-in user's customer profile
    wallet, created = Wallet.objects.get_or_create(customer=customer)  # Get or create a wallet for this customer
    transactions = Transaction.objects.filter(wallet=wallet)
    # transactions = Transaction.objects.filter(wallet=wallet).order_by('-date_added')

    return render(request, 'wallet.html', {'wallet': wallet, 'transactions': transactions})


@login_required
def add_amount(request):
    if request.method == "POST":
        amount = float(request.POST.get("amount", 0))

        if amount > 0:
            customer = Customer.objects.get(user=request.user)  # Get logged-in user's customer profile
            wallet, created = Wallet.objects.get_or_create(customer=customer)  # Get wallet for this customer
            wallet.amount += amount
            wallet.save()

            # Save transaction
            Transaction.objects.create(wallet=wallet, amount=amount)

    return redirect(wallet_view)


@login_required
def billing_session(request):
    customer = Customer.objects.get(user=request.user)  # Get logged-in user's customer profile
    wallet, created = Wallet.objects.get_or_create(customer=customer)  # Get wallet for this customer
    wallet_amount = wallet.amount

    total_price = request.session.get("total_price", 0)

    return render(request, "billing_session.html", {"total_price": total_price, "wallet_amount": wallet_amount})


@login_required
def pay_now(request):
    customer = Customer.objects.get(user=request.user)  # Get logged-in user's customer profile
    total_price = request.session.get("total_price", 0)  # Get the total price from session

    if request.method == "POST":
        wallet, created = Wallet.objects.get_or_create(customer=customer)  # Get or create wallet for the customer

        if wallet.amount >= total_price:
            # Deduct the amount from the wallet
            wallet.amount -= total_price
            wallet.save()

            # Log the transaction
            Transaction.objects.create(wallet=wallet, amount=-total_price)

            # Clear session total_price after payment
            del request.session["total_price"]

            # Render the payment success page
            return redirect(payment_success)
        else:
            # Insufficient funds message
            messages.error(request, "Insufficient balance in wallet. Please add funds.")

    return render(request, "pay_now.html", {
        "customer_name": customer.user.first_name,
        "customer_id": customer.customer_id,
        "total_price": total_price,
    })


def payment_success(request):
    return render(request, "payment_success.html")
