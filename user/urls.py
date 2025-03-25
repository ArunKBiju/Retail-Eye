from django.urls import path
from .views import *
urlpatterns = [
    path('', home, name='home'),
    path('dashboard/<int:customer_id>/', dashboard, name='dashboard'),
    path('registration', registration, name='registration'),
    path('face_image/', face_image, name='face_image'),
    path('user_login/', user_login, name='user_login'),
    path('face_identity/', face_identity, name='face_identity'),
    path('user_logout/', user_logout, name='user_logout'),
    path('otp_generate/', otp_generate, name='otp_generate'),
    path('password_reset/', password_reset, name='password_reset'),
    path('product_scan/', product_scan, name='product_scan'),
    path('billing_session/', billing_session, name='billing_session'),
    path('return_product/', return_product, name='return_product'),
    path('view_bill/', view_bill, name='view_bill'),
    path('pay_now/', pay_now, name='pay_now'),
    path('payment_success/', payment_success, name='payment_success'),
    path('wallet/', wallet_view, name='wallet'),
    path('add/', add_amount, name='add_amount'),
]
