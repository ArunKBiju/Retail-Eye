from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class Customer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    customer_id = models.IntegerField()


class Items(models.Model):
    item = models.CharField(max_length=200)
    price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


class Wallet(models.Model):
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE, related_name="wallet")
    amount = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.customer.user.username}'s Wallet - Balance: {self.amount} Rs"


class Transaction(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.CASCADE)
    amount = models.FloatField()
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.wallet.customer.user.username} - {self.amount} Rs on {self.date_added}"


