# streamlit_app.py
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import google.generativeai as genai

# Load the skin disease model
model = keras.models.load_model("best_skin_disease_model.h5")
img_height, img_width = 224, 224

# Class labels
class_labels = {
    0: 'BA Cellulitis',
    1: 'Impetigo',
    2: 'Athlete\'s Foot',
    3: 'Nail Fungus',
    4: 'Ringworm',
    5: 'Cutaneous Larva Migraines',
    6: 'Chicken Pox',
    7: 'Shingles'
}

# Disease info dictionary (same as before — shortened here for clarity)
disease_info = {
    'BA Cellulitis': {
        "symptoms": "Red, swollen, and tender skin, sometimes with fever and chills.",
        "causes": "Caused by bacteria like Streptococcus or Staphylococcus entering through a break in the skin.",
        "severity": "Moderate to severe if untreated; can spread quickly.",
        "treatment": "Oral or IV antibiotics, along with rest and hydration.",
        "precautions": "Keep skin clean, treat cuts and scrapes promptly, and monitor for any signs of infection.",
        "next_steps": "Consult a healthcare provider for antibiotics to prevent spread and complications."
    },
    'Impetigo': {
        "symptoms": "Red sores or blisters that burst and ooze, leaving a yellowish crust, commonly on face and hands.",
        "causes": "Bacterial infection, often caused by Staphylococcus or Streptococcus, and spread through contact.",
        "severity": "Mild to moderate; usually treatable with topical antibiotics.",
        "treatment": "Topical or oral antibiotics; hygiene to prevent spread.",
        "precautions": "Avoid touching or scratching sores; wash hands frequently to prevent spread to others.",
        "next_steps": "Apply prescribed creams, keep the affected areas clean, and avoid contact with others."
    },
    'Athlete\'s Foot': {
        "symptoms": "Itching, burning, and cracked or peeling skin, especially between toes.",
        "causes": "Fungal infection from warm, moist environments, such as locker rooms and public showers.",
        "severity": "Mild; can worsen if untreated but usually not serious.",
        "treatment": "Antifungal creams, powders, or oral antifungal medication if severe.",
        "precautions": "Avoid walking barefoot in communal areas; keep feet clean and dry.",
        "next_steps": "Apply antifungal treatment and maintain foot hygiene to prevent recurrence."
    },
    'Nail Fungus': {
        "symptoms": "Thickened, discolored, and brittle nails, often yellow or white.",
        "causes": "Fungal infection, often due to poor nail hygiene or contact with infected surfaces.",
        "severity": "Mild but can become chronic if untreated.",
        "treatment": "Topical antifungals or oral antifungal medications for severe cases.",
        "precautions": "Keep nails trimmed and dry; avoid sharing nail tools.",
        "next_steps": "Apply antifungal treatments as prescribed and consult a healthcare provider if it persists."
    },
    'Ringworm': {
        "symptoms": "Circular rash with a clearer center, itchy and scaly skin, often on body or scalp.",
        "causes": "Fungal infection spread by direct or indirect contact with an infected person or animal.",
        "severity": "Mild but highly contagious if untreated.",
        "treatment": "Antifungal creams for skin; oral antifungals for scalp or severe infections.",
        "precautions": "Avoid close contact with infected areas, keep skin clean and dry.",
        "next_steps": "Use antifungal treatment and consult a doctor if symptoms persist or worsen."
    },
    'Cutaneous Larva Migraines': {
        "symptoms": "Itchy, winding rash that appears under the skin, often on feet and legs.",
        "causes": "Parasitic infection from hookworm larvae found in contaminated soil.",
        "severity": "Usually mild but can be uncomfortable and persistent.",
        "treatment": "Antiparasitic medication, often albendazole or ivermectin.",
        "precautions": "Avoid walking barefoot in areas with potentially contaminated soil or sand.",
        "next_steps": "Consult a healthcare provider for antiparasitic treatment if the rash appears."
    },
    'Chicken Pox': {
        "symptoms": "Itchy, blister-like rash covering the body, along with fever and fatigue.",
        "causes": "Viral infection caused by the varicella-zoster virus, highly contagious.",
        "severity": "Mild in children; can be severe in adults or immunocompromised individuals.",
        "treatment": "Antihistamines for itching, antiviral medication if severe or at high risk.",
        "precautions": "Avoid contact with others during infection; vaccination available for prevention.",
        "next_steps": "Stay hydrated, rest, and monitor for complications, especially in adults."
    },
    'Shingles': {
        "symptoms": "Painful, blistering rash typically on one side of the body, with tingling or burning sensation.",
        "causes": "Reactivation of the varicella-zoster virus (the same virus that causes chicken pox).",
        "severity": "Moderate to severe, especially in older adults or immunocompromised individuals.",
        "treatment": "Antiviral medications, pain relief, and soothing lotions for rash.",
        "precautions": "Avoid contact with unvaccinated or immunocompromised individuals.",
        "next_steps": "See a healthcare provider for antiviral treatment; vaccinations are available for prevention."
    }
}
# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAOwdoSI99J_I0erHcgTnhP5W2JPl_OjBg"  # Use your real key
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("models/gemini-1.5-pro")

# Image preprocessing
def prepare_image(img):
    img = img.resize((img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# App title
st.set_page_config(layout="wide")
st.title("AI DERMA")

# Layout: Two columns side-by-side
left_col, right_col = st.columns([2, 1])
predicted_label = None

with left_col:
    st.header("Upload an Image to Detect Skin Disease")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            img_array = prepare_image(img)
            predictions = model.predict(img_array)
            predicted_index = int(np.argmax(predictions))
            predicted_label = class_labels.get(predicted_index, "Unknown")
            confidence = float(np.max(predictions))

            st.success(f"Prediction: {predicted_label}")
            st.info(f"Confidence: {confidence:.2f}")

            # Show disease details
            info = disease_info.get(predicted_label)
            if info:
                st.subheader("Disease Details")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.markdown(f"**Causes:** {info['causes']}")
                st.markdown(f"**Severity:** {info['severity']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
                st.markdown(f"**Precautions:** {info['precautions']}")
                st.markdown(f"**Next Steps:** {info['next_steps']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

with right_col:
    st.header("💬 Chatbot")

    # Automatically generate a prompt based on detected disease
    default_prompt = ""
    if predicted_label:
        default_prompt = f"Tell me more about {predicted_label}"

    user_message = st.text_area("Ask something or get more info:", value=default_prompt, height=150)

    if st.button("Ask Gemini"):
        if user_message.strip():
            try:
                response = model_gemini.generate_content(user_message)
                st.success("Gemini's Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message.")

