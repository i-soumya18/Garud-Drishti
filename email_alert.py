import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
def send_email(subject, message, sender_email, sender_password, receiver_email):
    # Create a connection to the email server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    # Create a message object and set the sender, receiver, subject and body
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Send the message and close the connection
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
