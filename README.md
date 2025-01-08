# Custodia

## Autors
- **Gerard Gispert Carreras**
- **Carles Lobon Quilez**
- **Miquel Garcia Bardaji**
- **Gerard Farré Uroz**

## Descripció
Aquest projecte implementa una aplicació que gestiona la ciberseguretat en 2 sistemes: Intrusion Detection System (IDS) i un detector de phishing per a missatges (mails per el moment, missatgeria instantània en un futur). És una solució pensada per concenciar i protegir les PIMES davant dels ciberatacs que poden rebre d'una manera fàcil i entenedora d'utilitzar, a baix cost i sense recursos tècnics (com personal) afegits. En conclusió, una eina per a tothom.

El projecte està desenvolupat utilitzant Python amb una llicència de codi obert sota els termes de l'[Apache License 2.0](LICENSE).

## Comentaris sobre el codi
- **Estructura:**  
  El projecte té com a principal arxiu l'app.py, fitxer programat amb el framework streamlit que ha servit per la demo. Cada sistema està programat per separat.

- **IDS:**  
  El Intrusion Detection System es basa en una xarxa neuronal que analitza el tràfic de xarxa resumit en netflow. 

  El model ha estat entrenat amb un dataset que inclou tots els atacs. Estan les llibretes de python en el directori models/
  A part, també està integrada amb el mail els reports que es generen dels anàlisis.

- **Phishing detector:**  
  El phishing detector s'ha aconseguit seguint l'estudi: [ChatSpamDetector](https://arxiv.org/pdf/2402.18093).

  Es troba separat en 2 fitxers. El Prompter és l'encarregat de preparar el mails com a prompt per LLM i el SpamDetector és el que connecta amb el LLM que detecta el phishing, Gemini en aquest moment, però està fet així perquè canviar el model sigui molt senzill.

## Llicència
Aquest projecte està llicenciat sota l'[Apache License 2.0](LICENSE).
