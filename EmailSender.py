import smtplib
from email.message import EmailMessage
import pandas as pd

class EmailSender:
    def __init__(self):
        self.SMTP_SERVER = "smtp.gmail.com"  # Servidor SMTP (Gmail)
        self.SMTP_PORT = 587  # Puerto SMTP
        self.EMAIL_ADDRESS = "nids.custodia@gmail.com"  
        self.EMAIL_PASSWORD = "hcfa bxyd msdp spoc" #"cfpc vgzy itai hsyn"  # Reemplaza con tu contraseña


    # Función para enviar el correo
    def __enviar_correo(self, destinatario, asunto, contenido):
        try:
            # Crear el mensaje de correo
            mensaje = EmailMessage()
            mensaje["From"] = self.EMAIL_ADDRESS
            mensaje["To"] = destinatario
            mensaje["Subject"] = asunto
            mensaje.set_content(contenido)

            # Conectar al servidor SMTP
            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as servidor:
                servidor.starttls()  
                servidor.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)  
                servidor.send_message(mensaje)  
            print("Correo enviado exitosamente.")
        except Exception as e:
            print(f"Error al enviar el correo: {e}")

    
    def sendEmail(self, labels, src_IPs, dst_IPs):
        condicion = True  
        if condicion:
            destinatario = "gerard.farre.uroz@estudiantat.upc.edu"
            asunto = "Notification: Threats have been detected on your network"
            contenido = "The following attacks have been detected:" + "\n\n" + "Attack: src_IP  dst_IP" + "\n"
            
            df = pd.DataFrame({"Label": labels, "Src_IP": src_IPs, "Dst_IP": dst_IPs})

            filtered_df = df[df["Label"] != "Benign"]

            filtered_df.loc[:, "Text"] = filtered_df["Label"] + ":  " + filtered_df["Src_IP"] + "  " + filtered_df["Dst_IP"]

            texto = "\n".join(filtered_df["Text"])
            contenido = contenido + texto
            self.__enviar_correo(destinatario, asunto, contenido)


