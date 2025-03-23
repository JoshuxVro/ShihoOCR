import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
import sounddevice as sd
import soundfile as sf
import uuid
from io import BytesIO
import re
from datetime import datetime, timedelta
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Set Tesseract path explicitly - modify this to your installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create directories for saving data if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("voice_notes", exist_ok=True)

# Load spaCy model for NLP tasks (if not installed, run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Function to perform OCR on image
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

# Advanced function to prioritize text
def prioritize_text(text, user_preferences=None):
    if not text.strip():
        return 0, {"urgency": 0, "deadline_proximity": 0, "named_entities": 0, "task_indicators": 0, "length_quality": 0, "user_relevance": 0}
    
    # Initialize score components
    scores = {
        "urgency": 0,
        "deadline_proximity": 0,
        "named_entities": 0,
        "task_indicators": 0,
        "length_quality": 0,
        "user_relevance": 0
    }
    
    # Process text with spaCy for advanced NLP analysis
    doc = nlp(text)
    
    # 1. Urgency scoring - keywords with weighted importance
    urgency_keywords = {
        "critical": 10, "urgent": 10, "emergency": 10, "asap": 9, "immediately": 9,
        "important": 8, "priority": 8, "deadline": 7, "due": 7, "required": 6,
        "needed": 6, "soon": 5, "quickly": 5, "timely": 5, "expedite": 5
    }
    
    text_lower = text.lower()
    for keyword, weight in urgency_keywords.items():
        if keyword in text_lower:
            # Additional points if keyword appears at beginning
            if re.search(r'\b' + keyword + r'\b', text_lower[:100]):
                scores["urgency"] += weight * 1.5
            else:
                scores["urgency"] += weight
    
    # Cap urgency score
    scores["urgency"] = min(scores["urgency"], 30)
    
    # 2. Deadline extraction and proximity scoring
    date_patterns = [
        # MM/DD/YYYY or DD/MM/YYYY
        r'\b(0?[1-9]|1[0-2])[/.-](0?[1-9]|[12][0-9]|3[01])[/.-](20\d{2}|\d{2})\b',
        # Month name formats
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (0?[1-9]|[12][0-9]|3[01])(st|nd|rd|th)?,? ?(20\d{2}|\d{2})?\b',
        # Tomorrow, next week, etc.
        r'\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week)\b'
    ]
    
    today = datetime.now()
    deadline_found = False
    closest_deadline = None
    
    # Check for explicit date patterns
    for pattern in date_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            match_text = match.group().lower()
            
            # Variable to store parsed date
            date = None
            
            # Convert relative dates to actual dates
            if "today" in match_text:
                date = today
            elif "tomorrow" in match_text:
                date = today + timedelta(days=1)
            elif "next week" in match_text:
                date = today + timedelta(days=7)
            elif any(day in match_text for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                # Simple mapping - would need improvement for real usage
                day_to_num = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                for day, num in day_to_num.items():
                    if day in match_text:
                        days_ahead = (num - today.weekday()) % 7
                        if days_ahead == 0:  # Today
                            days_ahead = 7  # Next week
                        date = today + timedelta(days=days_ahead)
                        break
            else:
                # Try to parse standard date format (simplified)
                try:
                    date_str = match_text
                    # Very simplified parsing - would need improvement
                    if re.match(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', date_str):
                        parts = re.split(r'[/.-]', date_str)
                        if len(parts) == 3:
                            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                            if year < 100:
                                year += 2000
                            date = datetime(year, month, day)
                except:
                    # If parsing fails, continue to next match
                    continue
            
            # Only update if a valid date was parsed
            if date is not None:
                deadline_found = True
                # Update closest deadline
                if closest_deadline is None or date < closest_deadline:
                    closest_deadline = date
    
    # Score based on deadline proximity
    if deadline_found and closest_deadline:
        days_until = (closest_deadline - today).days
        if days_until < 0:  # Past deadline
            scores["deadline_proximity"] = 30  # Highest priority for past due
        elif days_until == 0:  # Due today
            scores["deadline_proximity"] = 25
        elif days_until == 1:  # Due tomorrow
            scores["deadline_proximity"] = 20
        elif days_until <= 3:  # Due within 3 days
            scores["deadline_proximity"] = 15
        elif days_until <= 7:  # Due within a week
            scores["deadline_proximity"] = 10
        elif days_until <= 14:  # Due within two weeks
            scores["deadline_proximity"] = 5
        else:
            scores["deadline_proximity"] = 2
    
    # 3. Named entity recognition for importance
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # More points for people and organizations
    person_count = sum(1 for _, label in entities if label == "PERSON")
    org_count = sum(1 for _, label in entities if label == "ORG")
    
    scores["named_entities"] = min(person_count * 2 + org_count * 1.5, 15)
    
    # 4. Task indicators and action items
    task_indicators = [
        r"\bplease\b.*?\b(do|review|send|prepare|create|make|finish)\b",
        r"\bneed\b.*?\b(to|for)\b",
        r"\b(task|action item|to-?do|assignment)\b",
        r"\byou\b.*?\b(must|should|have to|need to)\b",
        r"\b(review|complete|send|submit|prepare|deliver)\b.*?\bby\b"
    ]
    
    action_count = 0
    for pattern in task_indicators:
        action_count += len(re.findall(pattern, text_lower))
    
    scores["task_indicators"] = min(action_count * 3, 15)
    
    # 5. Text quality, length and structure
    word_count = len(re.findall(r'\b\w+\b', text))
    
    # Favor medium-length texts (not too short, not too long)
    if word_count < 10:
        length_score = word_count / 2  # Very short gets low score
    elif 10 <= word_count <= 200:
        length_score = 10 * (word_count / 200)  # Steadily increases
    else:
        length_score = 10 - min(word_count - 200, 300) / 100  # Decreases for very long texts
    
    # Check for structured content (bullets, numbers)
    if re.search(r'(\n\s*[-•*]\s|\n\s*\d+\.|\n\s*\(?\d+\)?\.?)', text):
        length_score *= 1.5  # Bonus for structured lists
    
    scores["length_quality"] = min(length_score, 10)
    
    # 6. User relevance (based on user preferences if available)
    if user_preferences:
        # Simple relevance based on key terms important to the user
        relevant_terms = user_preferences.get("relevant_terms", [])
        if relevant_terms:
            relevance_count = sum(1 for term in relevant_terms if term.lower() in text_lower)
            scores["user_relevance"] = min(relevance_count * 3, 10)
    
    # Calculate final score (weighted combination)
    weights = {
        "urgency": 0.25,
        "deadline_proximity": 0.25,
        "named_entities": 0.15,
        "task_indicators": 0.15,
        "length_quality": 0.1,
        "user_relevance": 0.1
    }
    
    final_score = sum(score * weights[component] for component, score in scores.items())
    
    # Scale to 0-10
    normalized_score = min(final_score / 10, 10)
    
    # Return both the final score and component scores for explanation
    return normalized_score, scores

# Function to record voice note
def record_voice_note(duration=5):
    st.write("Recording...")
    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    progress_bar = st.progress(0)
    
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    
    sd.wait()
    st.write("Recording complete!")
    
    # Generate filename
    filename = f"voice_notes/voice_{str(uuid.uuid4())[:8]}.wav"
    sf.write(filename, recording, sample_rate)
    
    return filename

# Function to extract keywords for auto-tagging
def extract_keywords(text, top_n=5):
    if not text or len(text.strip()) < 10:
        return []
    
    doc = nlp(text)
    
    # Remove stopwords and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    
    # Count word frequencies
    word_freq = Counter(tokens)
    
    # Get the most common words
    common_words = word_freq.most_common(top_n)
    
    # Filter out single characters and return only words that appear at least twice
    keywords = [word for word, count in common_words if len(word) > 1 and count > 1]
    
    # Add named entities as keywords
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART"]]
    
    # Combine and return unique keywords
    all_keywords = list(set(keywords + entities))
    return all_keywords[:top_n]

# App title and description
st.title("AI Secretary: Screenshot Processor")
st.write("Upload screenshots, extract text, prioritize, and add voice notes")

# Load user preferences from file or initialize defaults
user_prefs_file = "user_preferences.json"
if os.path.exists(user_prefs_file):
    user_preferences = pd.read_json(user_prefs_file, typ="series").to_dict()
else:
    user_preferences = {
        "relevant_terms": ["project", "meeting", "report", "client", "deadline"],
        "priority_threshold": 7.0  # High priority threshold
    }

# Add Tesseract version check
try:
    tesseract_version = pytesseract.get_tesseract_version()
    st.sidebar.success(f"Tesseract found: v{tesseract_version}")
except Exception as e:
    st.sidebar.error(f"Tesseract not properly configured: {e}")
    st.sidebar.info("Please check the Tesseract path in the code.")
    st.sidebar.code("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")

# Sidebar for app navigation
page = st.sidebar.selectbox("Navigation", ["Upload & Process", "View Saved Items", "Settings"])

if page == "Upload & Process":
    # File uploader for screenshots
    uploaded_files = st.file_uploader("Upload screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process each uploaded file
        all_results = []
        
        for uploaded_file in uploaded_files:
            # Display progress
            st.write(f"Processing: {uploaded_file.name}")
            
            # Open image
            image = Image.open(uploaded_file)
            
            # Display image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=uploaded_file.name, width=300)
            
            # Perform OCR
            with st.spinner("Performing OCR..."):
                extracted_text = perform_ocr(image)
            
            # Calculate priority score with component breakdown
            priority_score, score_components = prioritize_text(extracted_text, user_preferences)
            
            # Auto-suggest tags based on content
            suggested_tags = extract_keywords(extracted_text)
            
            # Save image
            img_filename = f"uploads/{str(uuid.uuid4())[:8]}_{uploaded_file.name}"
            image.save(img_filename)
            
            with col2:
                st.write("Extracted Text:")
                st.text_area("", extracted_text, height=150, key=f"text_{uploaded_file.name}")
                
                # Priority visualization
                st.write(f"**Priority Score: {priority_score:.2f}/10**")
                
                # Show what contributed to the score
                with st.expander("Priority Score Breakdown"):
                    for component, score in score_components.items():
                        st.write(f"**{component.replace('_', ' ').title()}**: {score:.2f}")
                
                # Add tags with suggestions
                suggested_tags_str = ", ".join(suggested_tags) if suggested_tags else ""
                tags = st.text_input("Add tags (comma separated)", 
                                   value=suggested_tags_str,
                                   key=f"tags_{uploaded_file.name}")
                
                # Voice note option
                if st.button("Record Voice Note", key=f"rec_{uploaded_file.name}"):
                    duration = st.slider("Recording duration (seconds)", 3, 15, 5, key=f"dur_{uploaded_file.name}")
                    voice_filename = record_voice_note(duration)
                    st.audio(voice_filename)
                else:
                    voice_filename = None
            
            # Add to results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_results.append({
                "filename": img_filename,
                "text": extracted_text,
                "priority_score": priority_score,
                "urgency_score": score_components["urgency"],
                "deadline_score": score_components["deadline_proximity"],
                "entities_score": score_components["named_entities"],
                "task_score": score_components["task_indicators"],
                "quality_score": score_components["length_quality"],
                "relevance_score": score_components["user_relevance"],
                "tags": tags,
                "voice_note": voice_filename,
                "timestamp": timestamp
            })
            
            st.markdown("---")
        
        # Save results to CSV if there are new items
        if all_results:
            # Load existing data if available
            csv_file = "screenshot_data.csv"
            if os.path.exists(csv_file):
                existing_data = pd.read_csv(csv_file)
                updated_data = pd.concat([existing_data, pd.DataFrame(all_results)], ignore_index=True)
            else:
                updated_data = pd.DataFrame(all_results)
            
            updated_data.to_csv(csv_file, index=False)
            st.success(f"Processed {len(all_results)} screenshots and saved data!")

elif page == "View Saved Items":
    csv_file = "screenshot_data.csv"
    
    if not os.path.exists(csv_file):
        st.warning("No saved screenshots found. Process some screenshots first!")
    else:
        # Load data
        data = pd.read_csv(csv_file)
        
        # Sorting options
        sort_option = st.selectbox("Sort by", ["Priority (High to Low)", "Priority (Low to High)", 
                                             "Date (Newest)", "Date (Oldest)",
                                             "Urgency", "Deadline Proximity"])
        
        if sort_option == "Priority (High to Low)":
            data = data.sort_values(by="priority_score", ascending=False)
        elif sort_option == "Priority (Low to High)":
            data = data.sort_values(by="priority_score", ascending=True)
        elif sort_option == "Date (Newest)":
            data = data.sort_values(by="timestamp", ascending=False)
        elif sort_option == "Date (Oldest)":
            data = data.sort_values(by="timestamp", ascending=True)
        elif sort_option == "Urgency":
            data = data.sort_values(by="urgency_score", ascending=False)
        elif sort_option == "Deadline Proximity":
            data = data.sort_values(by="deadline_score", ascending=False)
        
        # Priority threshold filter
        show_high_priority = st.checkbox("Show only high priority items", value=False)
        if show_high_priority:
            threshold = user_preferences.get("priority_threshold", 7.0)
            data = data[data["priority_score"] >= threshold]
        
        # Filter by tags
        all_tags = []
        for tags_str in data["tags"].dropna():
            all_tags.extend([tag.strip() for tag in tags_str.split(",")])
        unique_tags = list(set([tag for tag in all_tags if tag]))
        
        if unique_tags:
            selected_tag = st.selectbox("Filter by tag", ["All"] + unique_tags)
            if selected_tag != "All":
                data = data[data["tags"].str.contains(selected_tag, na=False)]
        
        # Search by text content
        search_query = st.text_input("Search in text content")
        if search_query:
            data = data[data["text"].str.contains(search_query, case=False, na=False)]
        
        # Display items
        st.write(f"Showing {len(data)} items")
        
        for _, row in data.iterrows():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    if os.path.exists(row["filename"]):
                        st.image(row["filename"], width=200)
                    else:
                        st.warning("Image file not found")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            with col2:
                # Priority badge
                priority = row['priority_score']
                if priority >= 8:
                    st.markdown(f"<span style='background-color:red;color:white;padding:2px 8px;border-radius:10px'>Urgent: {priority:.1f}/10</span>", unsafe_allow_html=True)
                elif priority >= 5:
                    st.markdown(f"<span style='background-color:orange;color:white;padding:2px 8px;border-radius:10px'>Important: {priority:.1f}/10</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='background-color:green;color:white;padding:2px 8px;border-radius:10px'>Normal: {priority:.1f}/10</span>", unsafe_allow_html=True)
                
                st.write(f"Tags: {row['tags']}")
                st.write(f"Date: {row['timestamp']}")
                
                with st.expander("Show Extracted Text"):
                    st.write(row["text"])
                
                if "urgency_score" in row:
                    with st.expander("Priority Score Breakdown"):
                        components = {
                            "Urgency": row.get("urgency_score", 0),
                            "Deadline": row.get("deadline_score", 0),
                            "Entities": row.get("entities_score", 0),
                            "Task Indicators": row.get("task_score", 0),
                            "Content Quality": row.get("quality_score", 0),
                            "User Relevance": row.get("relevance_score", 0)
                        }
                        
                        for name, value in components.items():
                            st.write(f"**{name}**: {value:.1f}")
                
                if pd.notna(row["voice_note"]) and os.path.exists(row["voice_note"]):
                    st.audio(row["voice_note"])
                
                # Add action buttons
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.button("Mark Complete", key=f"complete_{_}")
                with col_b:
                    st.button("Snooze", key=f"snooze_{_}")
                with col_c:
                    st.button("Delete", key=f"delete_{_}")
            
            st.markdown("---")

elif page == "Settings":
    st.header("AI Secretary Settings")
    
    st.subheader("Relevance Settings")
    # Edit relevant terms
    relevant_terms = user_preferences.get("relevant_terms", [])
    relevant_terms_str = ", ".join(relevant_terms)
    new_terms = st.text_area("Terms relevant to you (comma separated)", value=relevant_terms_str)
    user_preferences["relevant_terms"] = [term.strip() for term in new_terms.split(",") if term.strip()]
    
    # Priority threshold
    priority_threshold = user_preferences.get("priority_threshold", 7.0)
    new_threshold = st.slider("High priority threshold", 0.0, 10.0, float(priority_threshold), 0.5)
    user_preferences["priority_threshold"] = new_threshold
    
    # Save settings
    if st.button("Save Settings"):
        pd.Series(user_preferences).to_json(user_prefs_file)
        st.success("Settings saved successfully!")
    
    # Advanced settings section
    with st.expander("Advanced Settings"):
        st.write("These settings require app restart to take effect")
        tesseract_path = st.text_input("Tesseract Path", value=pytesseract.pytesseract.tesseract_cmd)
        st.code(f"pytesseract.pytesseract.tesseract_cmd = r'{tesseract_path}'")
        
        st.subheader("Data Management")
        if st.button("Clear All Data"):
            if os.path.exists("screenshot_data.csv"):
                os.remove("screenshot_data.csv")
            st.success("All data cleared!")

# Add information about dependencies
with st.sidebar.expander("App Information"):
    st.write("""
    This app requires:
    - pytesseract (and Tesseract OCR installed)
    - spacy (with en_core_web_sm model)
    - scikit-learn
    - sounddevice
    - soundfile
    - Other standard libraries
    
    Install required Python packages:
    ```
    pip install streamlit pytesseract pillow pandas numpy spacy scikit-learn sounddevice soundfile
    python -m spacy download en_core_web_sm
    ```
    """)