import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Kompetenzabgleich & Weiterbildungsempfehlungen",
    page_icon="",
    layout="wide"
)

def save_employees_to_csv(employees_data, filename='data/employees_data.csv'):
    """Speichert Mitarbeiterdaten in einer CSV-Datei"""
    try:
        employees_data.to_csv(filename, index=False)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern der Mitarbeiterdaten: {str(e)}")
        return False

def load_employees_from_csv(filename='data/employees_data.csv'):
    """Lädt Mitarbeiterdaten aus einer CSV-Datei"""
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
            required_columns = [
                'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
                'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
                'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
            ]
            # Füge fehlende Spalten hinzu
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            # Stelle sicher, dass die Spalten in der richtigen Reihenfolge sind
            df = df[required_columns]
            # Behandle NaN-Werte
            for col in ['Manual_Skills', 'ESCO_Role', 'KldB_5_digit', 'Name', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']:
                df[col] = df[col].fillna('')
            return df
        else:
            return pd.DataFrame(columns=[
                'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
                'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
                'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
            ])
    except Exception as e:
        st.error(f"Fehler beim Laden der Mitarbeiterdaten: {str(e)}")
        return pd.DataFrame(columns=[
            'Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role',
            'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label',
            'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'
        ])

def manual_csv_parser(file_path, skip_rows=0):
    """Manueller CSV-Parser für problematische Dateien"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Überspringe die ersten Zeilen
        lines = lines[skip_rows:]
        
        # Finde die längste Zeile um die Anzahl der Spalten zu bestimmen
        max_columns = 0
        for line in lines:
            if line.strip():
                columns = line.count(',') + 1
                max_columns = max(max_columns, columns)
        
        # Erstelle Spaltennamen
        columns = [f'col_{i}' for i in range(max_columns)]
        
        # Parse die Daten
        data = []
        for line in lines:
            if line.strip():
                # Teile die Zeile und stelle sicher, dass sie die richtige Anzahl Spalten hat
                parts = line.strip().split(',')
                if len(parts) < max_columns:
                    parts.extend([''] * (max_columns - len(parts)))
                elif len(parts) > max_columns:
                    parts = parts[:max_columns]
                data.append(parts)
        
        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        st.error(f"Manueller Parser fehlgeschlagen: {str(e)}")
        return pd.DataFrame()

def normalize_esco_code(esco_code):
    """Normalisiert ESCO-Codes für besseren Vergleich"""
    if pd.isna(esco_code):
        return ""
    
    esco_code = str(esco_code).strip()
    
    # Falls es eine vollständige URI ist, extrahiere den Code
    if esco_code.startswith('http://data.europa.eu/esco/occupation/'):
        # Extrahiere den Code aus der URI
        parts = esco_code.split('/')
        if len(parts) > 0:
            return parts[-1]  # Nimm den letzten Teil
    
    # Falls es ein UUID ist, behalte es so
    if len(esco_code) == 36 and '-' in esco_code:
        return esco_code
    
    # Falls es ein kurzer Code ist (z.B. C0110), behalte es so
    return esco_code

@st.cache_data
def load_data():
    """Lädt alle benötigten CSV-Dateien"""
    try:
        # KldB zu ESCO Mapping
        kldb_esco_df = pd.read_csv('data/KldB_to_ESCO_Mapping_clean.csv')
        st.success("KldB-ESCO Mapping geladen")
        
        # ESCO Beruf-Skill Beziehungen (die richtige Datei!)
        try:
            occupation_skill_relations_df = pd.read_csv('data/occupationSkillRelations_de.csv', on_bad_lines='skip')
            st.success("ESCO Beruf-Skill Beziehungen geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Beruf-Skill Beziehungen: {str(e)}")
            occupation_skill_relations_df = pd.DataFrame()
        
        # ESCO Berufe
        try:
            occupations_df = pd.read_csv('data/occupations_de.csv', on_bad_lines='skip')
            st.success("ESCO Berufe geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Berufe: {str(e)}")
            occupations_df = pd.DataFrame()
        
        # ESCO Skills (Deutsch)
        try:
            skills_df = pd.read_csv('data/skills_de.csv', on_bad_lines='skip')
            st.success("ESCO Skills (Deutsch) geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Skills (Deutsch): {str(e)}")
            skills_df = pd.DataFrame()
        
        # ESCO Skills (Englisch)
        try:
            skills_en_df = pd.read_csv('data/skills_en.csv', on_bad_lines='skip')
            st.success("ESCO Skills (Englisch) geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der ESCO Skills (Englisch): {str(e)}")
            skills_en_df = pd.DataFrame()
        
        # EURES Skills Mapping
        try:
            eures_skills_df = pd.read_csv('data/EURESmapping_skills_DE.csv', on_bad_lines='skip')
            st.success("EURES Skills Mapping geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden des EURES Skills Mappings: {str(e)}")
            eures_skills_df = pd.DataFrame()
        
        # Udemy Kurse
        try:
            udemy_courses_df = pd.read_csv('data/Udemy_Course_Desc.csv', on_bad_lines='skip')
            st.success("Udemy Kurse geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der Udemy Kurse: {str(e)}")
            udemy_courses_df = pd.DataFrame()
        
        # Mitarbeiterdaten - Lade zuerst aus employees_data.csv, dann Fallback auf employee_input.csv
        try:
            employees_df = load_employees_from_csv('data/employees_data.csv')
            if not employees_df.empty:
                st.success("Mitarbeiterdaten aus employees_data.csv geladen")
            else:
                # Fallback auf employee_input.csv
                employees_df = pd.read_csv('data/employee_input.csv')
                st.success("Mitarbeiterdaten aus employee_input.csv geladen")
        except Exception as e:
            st.error(f"Fehler beim Laden der Mitarbeiterdaten: {str(e)}")
            employees_df = pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role'])
        
        # Erstelle ESCO Beruf-Skill Mapping
        occupation_skills_mapping = get_all_occupation_skills_direct(occupation_skill_relations_df, skills_df)
        
        return (employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, 
                eures_skills_df, udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df)
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {str(e)}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

@st.cache_data
def get_all_occupation_skills_direct(occupation_skill_relations_df, skills_df):
    """Erstellt ein direktes Mapping aller Berufe zu ihren Skills"""
    if occupation_skill_relations_df.empty or skills_df.empty:
        return {}
    
    # Erstelle ein Mapping von Skill-URIs zu Skill-Labels
    skill_uri_to_label = {}
    for _, skill_row in skills_df.iterrows():
        skill_uri = str(skill_row.get('conceptUri', ''))
        skill_label = str(skill_row.get('preferredLabel', ''))
        if skill_uri and skill_label and not pd.isna(skill_uri) and not pd.isna(skill_label):
            skill_uri_to_label[skill_uri] = skill_label
    
    # Erstelle das Beruf-zu-Skills Mapping
    occupation_skills = {}
    
    for _, relation_row in occupation_skill_relations_df.iterrows():
        occupation_uri = str(relation_row.get('occupationUri', ''))
        skill_uri = str(relation_row.get('skillUri', ''))
        relation_type = str(relation_row.get('relationType', ''))
        skill_type = str(relation_row.get('skillType', ''))
        
        if occupation_uri and skill_uri and not pd.isna(occupation_uri) and not pd.isna(skill_uri):
            if occupation_uri not in occupation_skills:
                occupation_skills[occupation_uri] = []
            
            # Hole das Skill-Label
            skill_label = skill_uri_to_label.get(skill_uri, skill_uri)
            
            occupation_skills[occupation_uri].append({
                'skill_uri': skill_uri,
                'skill_label': skill_label,
                'relation_type': relation_type,
                'skill_type': skill_type,
                'is_essential': relation_type.lower() == 'essential'
            })
    
    return occupation_skills

def find_occupation_by_label(occupations_df, target_label):
    """Findet einen Beruf anhand seines Labels"""
    if occupations_df.empty:
        return None
    
    # Suche nach exaktem Match
    exact_match = occupations_df[
        occupations_df['preferredLabel'].astype(str).str.contains(target_label, case=False, na=False)
    ]
    
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Suche nach Teil-Match
    partial_match = occupations_df[
        occupations_df['preferredLabel'].astype(str).str.contains(target_label.split()[0], case=False, na=False)
    ]
    
    if not partial_match.empty:
        return partial_match.iloc[0]
    
    return None

def get_skills_for_occupation_simple(occupation_label, occupation_skills_mapping, occupations_df, skill_mapping_with_english=None):
    """Einfache Funktion um Skills für einen Beruf zu finden (mit deutschen und englischen Labels)"""
    
    # 1. Finde den Beruf in der occupations_df
    occupation = find_occupation_by_label(occupations_df, occupation_label)
    
    if occupation is None:
        return []
    
    # 2. Hole die Skills für diesen Beruf
    occupation_uri = str(occupation['conceptUri'])
    
    if occupation_uri in occupation_skills_mapping:
        skills = occupation_skills_mapping[occupation_uri]
        
        # 3. Erweitere Skills um englische Labels, falls verfügbar
        if skill_mapping_with_english:
            enhanced_skills = []
            for skill in skills:
                skill_uri = skill.get('skill_uri', '')
                enhanced_skill = skill.copy()
                
                # Füge englische Labels hinzu, falls verfügbar
                if skill_uri in skill_mapping_with_english:
                    english_label = skill_mapping_with_english[skill_uri]['english']
                    enhanced_skill['skill_label_english'] = english_label
                    enhanced_skill['skill_labels_combined'] = f"{skill['skill_label']} | {english_label}"
                else:
                    enhanced_skill['skill_label_english'] = skill['skill_label']
                    enhanced_skill['skill_labels_combined'] = skill['skill_label']
                
                enhanced_skills.append(enhanced_skill)
            
            return enhanced_skills
        
        return skills
    
    return []

@st.cache_data
def create_employee_profile(employee_id, kldb_code, manual_skills, kldb_esco_df, occupation_skill_relations_df, skills_df, occupation_skills_mapping, occupations_df, saved_esco_role=None, manual_essential_skills='', manual_optional_skills='', removed_skills=''):
    """Erstellt ein Kompetenzprofil für einen Mitarbeiter basierend auf seiner aktuellen Rolle"""
    
    # Stelle sicher, dass alle Parameter Strings sind
    employee_id = str(employee_id)
    kldb_code = str(kldb_code)
    manual_skills = str(manual_skills)
    manual_essential_skills = str(manual_essential_skills)
    manual_optional_skills = str(manual_optional_skills)
    removed_skills = str(removed_skills)
    
    # 1. Finde die AKTUELLE Rolle des Mitarbeiters
    if saved_esco_role and saved_esco_role.strip():
        # Verwende die gespeicherte ESCO-Rolle
        current_occupation = kldb_esco_df[
            (kldb_esco_df['KldB_Code'].astype(str) == kldb_code) & 
            (kldb_esco_df['ESCO_Label'].astype(str) == saved_esco_role)
        ]
        
        if current_occupation.empty:
            # Fallback: Suche nur nach KldB-Code
            current_occupation = kldb_esco_df[
                kldb_esco_df['KldB_Code'].astype(str) == kldb_code
            ]
    else:
        # Suche nach KldB-Code (Standard-Verhalten)
        current_occupation = kldb_esco_df[
            kldb_esco_df['KldB_Code'].astype(str) == kldb_code
        ]
    
    if current_occupation.empty:
        # Fallback: Suche nach ähnlichen KldB-Codes
        current_occupation = kldb_esco_df[
            kldb_esco_df['KldB_Code'].astype(str).str.contains(kldb_code, na=False)
        ]
    
    if current_occupation.empty:
        return None
    
    # 2. Nimm nur die erste/primäre Rolle des Mitarbeiters
    primary_role = current_occupation.iloc[0]
    esco_label = str(primary_role['ESCO_Label'])
    
    # 3. Hole die Skills für die AKTUELLE Rolle des Mitarbeiters
    current_role_skills = get_skills_for_occupation_simple(esco_label, occupation_skills_mapping, occupations_df)
    
    # 4. Verarbeite entfernte Skills
    removed_skills_list = [s.strip().lower() for s in removed_skills.split(';') if s.strip()]
    filtered_skills = []
    for skill in current_role_skills:
        if skill['skill_label'].lower() not in removed_skills_list:
            filtered_skills.append(skill)
    
    # 5. Füge manuelle Essential Skills hinzu
    manual_essential_list = [s.strip() for s in manual_essential_skills.split(';') if s.strip()]
    for skill in manual_essential_list:
        # Suche nach der entsprechenden ESCO-Skill-URI
        skill_uri = f"manual_essential_{skill.lower().replace(' ', '_')}"
        
        # Versuche die echte ESCO-URI zu finden
        for _, skill_row in skills_df.iterrows():
            if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                skill_uri = str(skill_row.get('conceptUri', skill_uri))
                break
        
        filtered_skills.append({
            'skill_uri': skill_uri,
            'skill_label': skill,
            'relation_type': 'manual_essential',
            'skill_type': 'manual_essential',
            'is_essential': True
        })
    
    # 6. Füge manuelle Optional Skills hinzu
    manual_optional_list = [s.strip() for s in manual_optional_skills.split(';') if s.strip()]
    for skill in manual_optional_list:
        # Suche nach der entsprechenden ESCO-Skill-URI
        skill_uri = f"manual_optional_{skill.lower().replace(' ', '_')}"
        
        # Versuche die echte ESCO-URI zu finden
        for _, skill_row in skills_df.iterrows():
            if str(skill_row.get('preferredLabel', '')).lower() == skill.lower():
                skill_uri = str(skill_row.get('conceptUri', skill_uri))
                break
        
        filtered_skills.append({
            'skill_uri': skill_uri,
            'skill_label': skill,
            'relation_type': 'manual_optional',
            'skill_type': 'manual_optional',
            'is_essential': False
        })
    
    # 7. Füge ursprüngliche manuelle Skills hinzu (für Kompatibilität)
    manual_skills_list = [s.strip() for s in manual_skills.split(';') if s.strip()]
    for skill in manual_skills_list:
        filtered_skills.append({
            'skill_uri': f"manual_{skill.lower().replace(' ', '_')}",
            'skill_label': skill,
            'relation_type': 'manual',
            'skill_type': 'manual',
            'is_essential': True  # Manuelle Skills als essential behandeln
        })
    
    return {
        'employee_id': employee_id,
        'kldb_code': kldb_code,
        'current_role': primary_role.to_dict(),
        'skills': filtered_skills,
        'manual_skills': manual_skills_list,
        'manual_essential_skills': manual_essential_list,
        'manual_optional_skills': manual_optional_list,
        'removed_skills': removed_skills_list
    }

@st.cache_data
def get_all_esco_occupations(kldb_esco_df):
    """Holt alle verfügbaren ESCO-Berufe"""
    unique_occupations = kldb_esco_df[['ESCO_Code', 'ESCO_Label']].drop_duplicates()
    return unique_occupations.to_dict('records')

@st.cache_data
def calculate_occupation_match(employee_profile, target_occupation, occupation_skill_relations_df, skills_df, occupation_skills_mapping, occupations_df):
    """Berechnet den Match zwischen Mitarbeiter (aktueller Rolle) und neuer Zielrolle"""
    
    if not employee_profile or not target_occupation:
        return None
    
    # Hole Skills der NEUEN Zielrolle
    target_esco_label = str(target_occupation.get('ESCO_Label', ''))
    target_role_skills = get_skills_for_occupation_simple(target_esco_label, occupation_skills_mapping, occupations_df)
    
    # Hole Skills der AKTUELLEN Rolle des Mitarbeiters
    current_role_skills = employee_profile['skills']
    
    # Vergleiche Skills: Was hat der Mitarbeiter vs. was braucht die neue Rolle
    current_skill_labels = [skill['skill_label'].lower() for skill in current_role_skills]
    target_skill_labels = [skill['skill_label'].lower() for skill in target_role_skills]
    
    # Berechne Matches (Skills die der Mitarbeiter bereits hat)
    matching_skills = []
    missing_skills = []
    
    for target_skill in target_role_skills:
        target_label = target_skill['skill_label'].lower()
        if target_label in current_skill_labels:
            matching_skills.append(target_skill)
        else:
            missing_skills.append(target_skill)
    
    # Berechne Prozentsätze
    total_target_skills = len(target_role_skills)
    match_count = len(matching_skills)
    
    if total_target_skills == 0:
        return {
            'match_percentage': 0,
            'weighted_fit_percentage': 0,
            'matching_skills': [],
            'missing_skills': [],
            'has_target_skills': False,
            'current_role': employee_profile.get('current_role', {}),
            'target_role': target_occupation
        }
    
    match_percentage = (match_count / total_target_skills) * 100
    
    # Berechne Weighted Fit (essential skills zählen doppelt)
    weighted_matches = 0
    weighted_total = 0
    
    for target_skill in target_role_skills:
        weight = 2 if target_skill['is_essential'] else 1
        weighted_total += weight
        
        if target_skill['skill_label'].lower() in current_skill_labels:
            weighted_matches += weight
    
    weighted_fit_percentage = (weighted_matches / weighted_total) * 100 if weighted_total > 0 else 0
    
    return {
        'match_percentage': match_percentage,
        'weighted_fit_percentage': weighted_fit_percentage,
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'has_target_skills': True,
        'current_role': employee_profile.get('current_role', {}),
        'target_role': target_occupation
    }

@st.cache_data
def preprocess_text(text):
    """Bereitet Text für Tokenisierung vor"""
    if pd.isna(text):
        return ""
    
    # Konvertiere zu String und Kleinbuchstaben
    text = str(text).lower()
    
    # Entferne Sonderzeichen
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Einfache Tokenisierung ohne NLTK
    tokens = text.split()
    
    # Einfache Stopwords (deutsche und englische)
    stop_words = {
        'der', 'die', 'das', 'und', 'oder', 'aber', 'für', 'mit', 'von', 'zu', 'in', 'auf', 'an', 'bei',
        'the', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 'in', 'on', 'at', 'by', 'is', 'are', 'was', 'were',
        'ein', 'eine', 'einer', 'eines', 'einem', 'einen', 'a', 'an', 'this', 'that', 'these', 'those'
    }
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

@st.cache_data
def find_udemy_courses_for_skills(missing_skills, udemy_courses_df, top_k=5):
    """Findet passende Udemy-Kurse für fehlende Skills (mit deutschen und englischen Labels)"""
    
    if not missing_skills or udemy_courses_df.empty:
        return []
    
    # Prüfe ob benötigte Spalten vorhanden sind
    required_columns = ['Title', 'Headline', 'Description', 'URL', 'Price', 'Language']
    missing_columns = [col for col in required_columns if col not in udemy_courses_df.columns]
    if missing_columns:
        st.warning(f"Fehlende Spalten in Udemy-Daten: {missing_columns}")
        return []
    
    # Bereite Kursdaten vor
    udemy_courses_df['processed_text'] = (
        udemy_courses_df['Title'].fillna('') + ' ' +
        udemy_courses_df['Headline'].fillna('') + ' ' +
        udemy_courses_df['Description'].fillna('')
    ).apply(preprocess_text)
    
    # TF-IDF Vektorisierung
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    course_vectors = vectorizer.fit_transform(udemy_courses_df['processed_text'])
    
    recommendations = []
    
    for skill in missing_skills:
        # Prüfe ob skill ein Dictionary ist (mit deutschen und englischen Labels)
        if isinstance(skill, dict):
            # Verwende das kombinierte Label oder das deutsche Label
            skill_text = skill.get('skill_labels_combined', skill.get('skill_label', str(skill)))
        else:
            # Fallback für String-Skills
            skill_text = str(skill)
        
        # Bereite Skill-Text vor
        skill_text = preprocess_text(skill_text)
        skill_vector = vectorizer.transform([skill_text])
        
        # Berechne Ähnlichkeiten
        similarities = cosine_similarity(skill_vector, course_vectors).flatten()
        
        # Top-K Kurse
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        skill_recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Mindestähnlichkeit
                course = udemy_courses_df.iloc[idx]
                
                # Bestimme den Skill-Namen für die Anzeige
                if isinstance(skill, dict):
                    display_skill = skill.get('skill_label', str(skill))
                else:
                    display_skill = str(skill)
                
                skill_recommendations.append({
                    'skill': display_skill,
                    'course_title': course['Title'],
                    'course_headline': course['Headline'],
                    'course_description': course['Description'][:200] + '...' if len(str(course['Description'])) > 200 else course['Description'],
                    'course_url': course['URL'],
                    'course_price': course['Price'],
                    'course_language': course['Language'],
                    'similarity_score': similarities[idx]
                })
        
        if skill_recommendations:
            recommendations.extend(skill_recommendations)
    
    return recommendations

@st.cache_data
def create_isco_esco_mapping(occupations_df):
    """Erstellt eine Mapping-Tabelle zwischen ISCO-Gruppen und ESCO-URIs"""
    if occupations_df.empty:
        return {}
    
    mapping = {}
    
    for _, row in occupations_df.iterrows():
        isco_group = str(row.get('iscoGroup', ''))
        concept_uri = str(row.get('conceptUri', ''))
        preferred_label = str(row.get('preferredLabel', ''))
        
        if isco_group and concept_uri and not pd.isna(isco_group) and not pd.isna(concept_uri):
            # Extrahiere UUID aus der URI
            uuid = concept_uri.split('/')[-1]
            
            # Erstelle Mapping für ISCO-Gruppe
            if isco_group not in mapping:
                mapping[isco_group] = []
            
            mapping[isco_group].append({
                'uuid': uuid,
                'uri': concept_uri,
                'label': preferred_label
            })
    
    return mapping

@st.cache_data
def create_skill_mapping_with_english(skills_df, skills_en_df):
    """Erstellt ein Mapping zwischen deutschen und englischen Skills basierend auf conceptUri"""
    if skills_df.empty or skills_en_df.empty:
        return {}
    
    skill_mapping = {}
    
    # Erstelle ein Mapping von conceptUri zu deutschen und englischen Labels
    for _, skill_row in skills_df.iterrows():
        concept_uri = str(skill_row.get('conceptUri', ''))
        german_label = str(skill_row.get('preferredLabel', ''))
        
        if concept_uri and german_label and not pd.isna(concept_uri) and not pd.isna(german_label):
            # Suche nach der englischen Entsprechung
            english_match = skills_en_df[skills_en_df['conceptUri'] == concept_uri]
            
            if not english_match.empty:
                english_label = str(english_match.iloc[0].get('preferredLabel', ''))
                if english_label and not pd.isna(english_label):
                    skill_mapping[concept_uri] = {
                        'german': german_label,
                        'english': english_label
                    }
    
    return skill_mapping

@st.cache_data
def create_kldb_isco_mapping(kldb_esco_df):
    """Erstellt eine Mapping-Tabelle zwischen ESCO-Codes und ISCO-Gruppen"""
    if kldb_esco_df.empty:
        return {}
    
    mapping = {}
    
    # Mapping zwischen ESCO-Codes und ISCO-Gruppen basierend auf der Struktur
    # C0110 -> 0110 (Militärberufe)
    # C0210 -> 0210 (Unteroffiziere)
    # etc.
    
    for _, row in kldb_esco_df.iterrows():
        esco_code = str(row.get('ESCO_Code', ''))
        esco_label = str(row.get('ESCO_Label', ''))
        
        if esco_code and not pd.isna(esco_code):
            # Extrahiere die numerische Komponente aus dem ESCO-Code
            if esco_code.startswith('C'):
                isco_group = esco_code[1:]  # Entferne das 'C' und nimm den Rest
                mapping[esco_code] = isco_group
    
    return mapping

@st.cache_data
def get_all_available_esco_skills(skills_df, skills_en_df=None):
    """Lädt alle verfügbaren ESCO-Skills für Dropdown-Auswahl"""
    if skills_df.empty:
        return []
    
    available_skills = []
    
    for _, skill_row in skills_df.iterrows():
        skill_uri = str(skill_row.get('conceptUri', ''))
        german_label = str(skill_row.get('preferredLabel', ''))
        
        if skill_uri and german_label and not pd.isna(skill_uri) and not pd.isna(german_label):
            skill_info = {
                'uri': skill_uri,
                'german_label': german_label,
                'english_label': german_label,  # Fallback
                'display_label': german_label
            }
            
            # Füge englische Labels hinzu, falls verfügbar
            if skills_en_df is not None and not skills_en_df.empty:
                english_match = skills_en_df[skills_en_df['conceptUri'] == skill_uri]
                if not english_match.empty:
                    english_label = str(english_match.iloc[0].get('preferredLabel', ''))
                    if english_label and not pd.isna(english_label):
                        skill_info['english_label'] = english_label
                        skill_info['display_label'] = f"{german_label} | {english_label}"
            
            available_skills.append(skill_info)
    
    # Sortiere nach deutschem Label
    available_skills.sort(key=lambda x: x['german_label'])
    
    return available_skills

def main():
    st.title("Kompetenzabgleich & Weiterbildungsempfehlungen")
    st.markdown("---")
    
    # Lade Daten
    with st.spinner("Lade Daten..."):
        employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df = load_data()
    
    if employees_df.empty and kldb_esco_df.empty:
        st.error("Fehler beim Laden der Daten. Bitte überprüfe die CSV-Dateien.")
        return
    
    # Erstelle das direkte Skills-Mapping
    st.session_state.occupation_skills_mapping = occupation_skills_mapping
    
    # Erstelle das Skill-Mapping mit englischen Entsprechungen
    skill_mapping_with_english = create_skill_mapping_with_english(skills_df, skills_en_df)
    st.session_state.skill_mapping_with_english = skill_mapping_with_english
    
    # Session State für Mitarbeiterdaten initialisieren, falls nicht vorhanden
    if 'employees_data' not in st.session_state:
        # Füge Name-Spalte hinzu, falls sie nicht existiert
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
    required_columns = ['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
    for col in required_columns:
        if col not in st.session_state.employees_data.columns:
            st.session_state.employees_data[col] = ''
    
    # Behandle NaN-Werte in den Zielrollen-Spalten und neuen Skill-Spalten
    for col in ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']:
        st.session_state.employees_data[col] = st.session_state.employees_data[col].fillna('')
    
    # Lade gespeicherte Zielrollen für den aktuellen Mitarbeiter
    if 'current_employee_id' in st.session_state:
        current_employee_id = st.session_state.current_employee_id
        employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == current_employee_id]
        
        if not employee_data.empty:
            employee_row = employee_data.iloc[0]
            target_kldb_code = employee_row.get('Target_KldB_Code', '')
            target_kldb_label = employee_row.get('Target_KldB_Label', '')
            target_esco_code = employee_row.get('Target_ESCO_Code', '')
            target_esco_label = employee_row.get('Target_ESCO_Label', '')
            
            # Wenn gespeicherte Zielrollen vorhanden sind, stelle sie wieder her
            if target_kldb_code and target_kldb_label and target_esco_code and target_esco_label:
                st.session_state.selected_target_role = {
                    'KldB_Code': target_kldb_code,
                    'KldB_Label': target_kldb_label,
                    'ESCO_Code': target_esco_code,
                    'ESCO_Label': target_esco_label
                }
                
                # Zeige Benachrichtigung über wiederhergestellte Zielrolle
                if 'target_role_restored' not in st.session_state:
                    st.session_state.target_role_restored = True
                    st.success(f"Gespeicherte Zielrolle für {employee_row.get('Name', current_employee_id)} wiederhergestellt: {target_kldb_label}")
                else:
                    # Zeige Benachrichtigung über wiederhergestellte Zielrolle beim Mitarbeiterwechsel
                    st.success(f"Gespeicherte Zielrolle für {employee_row.get('Name', current_employee_id)} wiederhergestellt: {target_kldb_label}")
    

    
    # Verwende aktualisierte Mitarbeiterdaten aus Session State
    employees_df = st.session_state.employees_data
    
    # Globale Mitarbeiterauswahl in der Sidebar
    st.sidebar.title("Navigation")
    
    # Mitarbeiterauswahl (global für alle Sektionen)
    if not employees_df.empty:
        st.sidebar.subheader("Mitarbeiter auswählen")
        employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in employees_df.iterrows()]
        
        # Initialisiere selected_employee falls nicht vorhanden
        if 'selected_employee' not in st.session_state:
            st.session_state.selected_employee = employee_options[0] if employee_options else None
        
        # Verwende die gespeicherte Auswahl oder den ersten Mitarbeiter
        default_index = 0
        if st.session_state.selected_employee in employee_options:
            default_index = employee_options.index(st.session_state.selected_employee)
        
        selected_employee_str = st.sidebar.selectbox(
            "Mitarbeiter:", 
            employee_options, 
            key="global_employee_select",
            index=default_index
        )
        
        # Speichere die Auswahl im Session State
        st.session_state.selected_employee = selected_employee_str
        
        # Extrahiere Employee ID für weitere Verwendung
        if selected_employee_str:
            new_employee_id = selected_employee_str.split(" - ")[0]
            
            # Prüfe ob sich der Mitarbeiter geändert hat
            if 'current_employee_id' not in st.session_state or st.session_state.current_employee_id != new_employee_id:
                st.session_state.current_employee_id = new_employee_id
                
                # Reset Benachrichtigungen für neuen Mitarbeiter
                if 'target_role_restored' in st.session_state:
                    del st.session_state.target_role_restored
                
                # Lade gespeicherte Zielrollen für den neuen Mitarbeiter
                employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == new_employee_id]
                
                if not employee_data.empty:
                    employee_row = employee_data.iloc[0]
                    target_kldb_code = employee_row.get('Target_KldB_Code', '')
                    target_kldb_label = employee_row.get('Target_KldB_Label', '')
                    target_esco_code = employee_row.get('Target_ESCO_Code', '')
                    target_esco_label = employee_row.get('Target_ESCO_Label', '')
                    
                    # Wenn gespeicherte Zielrollen vorhanden sind, stelle sie wieder her
                    if target_kldb_code and target_kldb_label and target_esco_code and target_esco_label:
                        st.session_state.selected_target_role = {
                            'KldB_Code': target_kldb_code,
                            'KldB_Label': target_kldb_label,
                            'ESCO_Code': target_esco_code,
                            'ESCO_Label': target_esco_label
                        }
                    else:
                        # Lösche gespeicherte Zielrolle falls keine vorhanden
                        if 'selected_target_role' in st.session_state:
                            del st.session_state.selected_target_role
    
    # Navigation
    page = st.sidebar.selectbox(
        "Wählen Sie eine Sektion:",
        ["Mitarbeiter-Kompetenzprofile", "Berufsabgleich", "Kursempfehlungen", "Gesamtübersicht", "Mitarbeiter-Verwaltung"]
    )
    
    # Zeige entsprechende Seite
    if page == "Mitarbeiter-Kompetenzprofile":
        show_employee_profiles(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df)
    elif page == "Berufsabgleich":
        show_occupation_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df)
    elif page == "Kursempfehlungen":
        show_course_recommendations(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df)
    elif page == "Gesamtübersicht":
        show_overview(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df)
    elif page == "Mitarbeiter-Verwaltung":
        show_employee_management(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df)

def show_employee_profiles(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df=None):
    st.header("Mitarbeiter-Kompetenzprofile")
    
    # Verwende aktualisierte Mitarbeiterdaten aus Session State
    employees_df = st.session_state.employees_data
    
    if employees_df.empty:
        st.warning("Keine Mitarbeiterdaten gefunden.")
        return
    
    # Verwende die globale Mitarbeiterauswahl
    if 'current_employee_id' not in st.session_state:
        st.info("Bitte wählen Sie einen Mitarbeiter in der Sidebar aus.")
        return
    
    employee_id = st.session_state.current_employee_id
    employee_data = employees_df[employees_df['Employee_ID'] == employee_id].iloc[0]
    
    st.subheader(f"Profil von {employee_data.get('Name', employee_id)}")
    
    # Hole aktuelle Daten aus Session State (nicht gecacht)
    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
    current_kldb = current_employee_data.get('KldB_5_digit', '')
    current_manual_skills = current_employee_data.get('Manual_Skills', '')
    current_esco_role = current_employee_data.get('ESCO_Role', '')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Rolle:**")
        
        # Zeige die gespeicherte ESCO-Rolle oder finde sie basierend auf KldB
        if current_esco_role:
            st.write(f"Code: {current_kldb}")
            st.write(f"ESCO-Rolle: {current_esco_role}")
            
            # Finde die KldB-Rolle basierend auf dem Code
            kldb_match = kldb_esco_df[kldb_esco_df['KldB_Code'] == current_kldb]
            if not kldb_match.empty:
                st.write(f"KldB-Rolle: {kldb_match.iloc[0]['KldB_Label']}")
            else:
                st.write("KldB-Rolle: Nicht gefunden")
        else:
            # Fallback: Finde die entsprechende ESCO-Rolle basierend auf KldB
            kldb_match = kldb_esco_df[kldb_esco_df['KldB_Code'] == current_kldb]
            if not kldb_match.empty:
                esco_role = kldb_match.iloc[0]
                st.write(f"Code: {current_kldb}")
                st.write(f"ESCO-Rolle: {esco_role['ESCO_Label']}")
                st.write(f"KldB-Rolle: {esco_role['KldB_Label']}")
            else:
                st.write(f"Code: {current_kldb}")
                st.write("ESCO-Rolle: Nicht gefunden")
                st.write("KldB-Rolle: Nicht gefunden")
    
    with col2:
        st.write("**Manuelle Skills:**")
        if current_manual_skills and pd.notna(current_manual_skills) and str(current_manual_skills).strip():
            for skill in str(current_manual_skills).split(';'):
                if skill.strip():  # Nur nicht-leere Skills anzeigen
                    st.write(f"• {skill.strip()}")
        else:
            st.write("Keine manuellen Skills")
    
    # Zeige ausgewählte Zielrolle, falls vorhanden
    if 'selected_target_role' in st.session_state:
        st.markdown("---")
        st.subheader("Ausgewählte Zielrolle")
        
        target_role = st.session_state.selected_target_role
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Zielrolle Details:**")
            st.write(f"• KldB-Code: {target_role['KldB_Code']}")
            st.write(f"• KldB-Rolle: {target_role['KldB_Label']}")
        
        with col2:
            st.write("**Zugehörige ESCO-Rolle:**")
            st.write(f"• ESCO-Rolle: {target_role['ESCO_Label']}")
            st.write(f"• ESCO-Code: {target_role['ESCO_Code']}")
        
        # Button zum Zurücksetzen der Zielrolle
        if st.button("Zielrolle zurücksetzen", type="secondary"):
            # Lösche die gespeicherte Zielrolle aus den Mitarbeiterdaten
            st.session_state.employees_data.loc[
                st.session_state.employees_data['Employee_ID'] == employee_id, 
                ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label']
            ] = ['', '', '', '']
            
            # Speichere in CSV
            if save_employees_to_csv(st.session_state.employees_data):
                # Lösche aus Session State
                if 'selected_target_role' in st.session_state:
                    del st.session_state.selected_target_role
                st.success("Zielrolle wurde zurückgesetzt und gespeichert!")
            else:
                st.warning("Zielrolle zurückgesetzt, aber Speichern fehlgeschlagen!")
            
            st.rerun()
    
    st.markdown("---")
    
    # Zugewiesene Skills anzeigen (nach der Rollenzuweisung)
    st.subheader("Zugewiesene Skills")
    
    # Prüfe ob eine aktuelle Rolle zugewiesen wurde
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie unten eine Rolle zu, um die zugehörigen Skills anzuzeigen.")
    else:
        # Hole zusätzliche Skill-Daten aus Session State
        current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
        current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
        current_removed_skills = current_employee_data.get('Removed_Skills', '')
        
        # Erstelle das aktuelle Profil
        profile = create_employee_profile(
            employee_id,
            current_kldb,
            current_manual_skills,
            kldb_esco_df,
            occupation_skill_relations_df,
            skills_df,
            st.session_state.occupation_skills_mapping,
            occupations_df,
            current_esco_role,  # Übergebe die gespeicherte ESCO-Rolle
            current_manual_essential_skills,
            current_manual_optional_skills,
            current_removed_skills
        )
        
        if profile:
            current_role = profile['current_role']
            st.write(f"**Anzahl Skills:** {len(profile['skills'])}")
            
            # Zeige Skills mit Legende
            if profile['skills']:
                # Legende für Skill-Farbpunkte
                st.markdown("**Skill-Legende:**")
                legend_col1, legend_col2 = st.columns(2)
                with legend_col1:
                    st.write("**Essential Skills** - Unverzichtbare Skills (zählen doppelt)")
                with legend_col2:
                    st.write("**Optional Skills** - Hilfreiche Skills (zählen einfach)")
                
                st.markdown("---")
                
                st.write("**Zugewiesene Skills:**")
                for skill in profile['skills']:
                    if skill.get('relation_type') == 'manual':
                        st.write(f"• {skill['skill_label']} (manuell)")
                    elif skill.get('relation_type') == 'manual_essential':
                        st.write(f"• {skill['skill_label']} (Essential)")
                    elif skill.get('relation_type') == 'manual_optional':
                        st.write(f"• {skill['skill_label']} (Optional)")
                    else:
                        essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                        st.write(f"• {skill['skill_label']}{essential_mark}")
            else:
                st.info("Keine Skills zugewiesen.")
        else:
            st.warning("Konnte kein Kompetenzprofil erstellen.")
    
    st.markdown("---")
    
    # Manuelle Rollenzuweisung
    st.markdown("---")
    st.subheader("Aktuelle Rolle manuell zuweisen")
    
    # Aus verfügbaren KldB-Rollen wählen
    st.write("**Aus verfügbaren KldB-Rollen wählen**")
    
    # Erstelle eine Dropdown-Box mit allen verfügbaren KldB-Rollen
    # Entferne Duplikate basierend auf KldB_Code UND KldB_Label
    available_kldb_roles = kldb_esco_df[['KldB_Code', 'KldB_Label']].drop_duplicates(subset=['KldB_Code', 'KldB_Label'])
    available_kldb_roles = available_kldb_roles.sort_values('KldB_Label')
    
    # Erstelle Optionen für die Dropdown-Box - kürzere, saubere Anzeige
    kldb_options = []
    seen_options = set()  # Verhindert Duplikate in der Anzeige
    
    for _, row in available_kldb_roles.iterrows():
        kldb_label = str(row['KldB_Label']).strip()
        kldb_code = str(row['KldB_Code']).strip()
        
        # Überspringe leere Einträge
        if not kldb_label or not kldb_code or kldb_label == 'nan' or kldb_code == 'nan':
            continue
        
        # Kürze lange Labels für bessere Lesbarkeit
        display_label = kldb_label
        if len(display_label) > 40:
            display_label = display_label[:37] + "..."
        
        # Erstelle saubere Option
        option = f"{display_label} | {kldb_code}"
        
        # Verhindere Duplikate in der Anzeige
        if option not in seen_options:
            kldb_options.append(option)
            seen_options.add(option)
    
    # Sortiere die Optionen alphabetisch
    kldb_options.sort()
    kldb_options.insert(0, "Bitte wählen Sie eine KldB-Rolle...")
    
    # Zeige Anzahl der verfügbaren Rollen
    st.write(f"**Verfügbare KldB-Rollen:** {len(kldb_options) - 1} Rollen")
    
    selected_kldb_role = st.selectbox("Wählen Sie eine KldB-Rolle:", kldb_options, help="Scrollen Sie durch die Liste oder tippen Sie den Anfang des Berufsnamens")
    
    if selected_kldb_role and selected_kldb_role != "Bitte wählen Sie eine KldB-Rolle...":
        # Extrahiere KldB-Code aus der Auswahl
        kldb_code = selected_kldb_role.split(" | ")[1]
        kldb_label = selected_kldb_role.split(" | ")[0]
        
        # Finde das vollständige Label aus den Originaldaten
        full_label = available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ]['KldB_Label'].iloc[0] if not available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ].empty else kldb_label
        
        # Zeige die vollständige Information an
        st.write(f"**Ausgewählte KldB-Rolle:** {full_label} ({kldb_code})")
        
        # Finde alle zugehörigen ESCO-Rollen für diese KldB-Rolle
        matching_roles = kldb_esco_df[kldb_esco_df['KldB_Code'] == kldb_code]
        
        if not matching_roles.empty:
            st.write(f"**Verfügbare ESCO-Rollen für '{full_label}':**")
            
            for idx, role in matching_roles.iterrows():
                esco_label = role['ESCO_Label']
                esco_code = role['ESCO_Code']
                
                # Hole Skills für diese ESCO-Rolle
                role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                
                with st.expander(f"{esco_label} ({esco_code})"):
                    if role_skills:
                        # Legende für Skill-Farbpunkte
                        st.markdown("**Skill-Legende:**")
                        legend_col1, legend_col2 = st.columns(2)
                        with legend_col1:
                            st.write("**Essential Skills** - Unverzichtbare Skills")
                        with legend_col2:
                            st.write("**Optional Skills** - Hilfreiche Skills")
                        
                        st.markdown("---")
                        
                        st.write("**Skills:**")
                        for skill in role_skills:
                            essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                            st.write(f"• {skill['skill_label']}{essential_mark}")
                        
                        # Button zum Übernehmen
                        if st.button(f"Als aktuelle Rolle übernehmen", key=f"assign_kldb_{idx}"):
                            # Aktualisiere den KldB-Code in den Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'KldB_5_digit'
                            ] = kldb_code
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'ESCO_Role'
                            ] = esco_label # Speichere die ESCO-Rolle
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Rolle '{esco_label}' wurde als aktuelle Rolle zugewiesen und gespeichert!")
                            else:
                                st.warning(f"Rolle zugewiesen, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    else:
                        st.write("Keine Skills für diese Rolle gefunden.")
        else:
            st.warning(f"Keine ESCO-Rollen für KldB-Rolle '{full_label}' gefunden.")
    
    # Option 3: Essential und Optional Skills manuell anpassen
    st.markdown("---")
    st.subheader("Essential und Optional Skills anpassen")
    
    # Prüfe ob eine aktuelle Rolle zugewiesen wurde
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie oben eine Rolle zu, um die Skills anzupassen.")
    else:
        # Hole aktuelle Skill-Daten
        current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
        current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
        current_removed_skills = current_employee_data.get('Removed_Skills', '')
        
        # Erstelle das aktuelle Profil für die Skill-Anzeige
        profile_for_skills = create_employee_profile(
            employee_id,
            current_kldb,
            current_manual_skills,
            kldb_esco_df,
            occupation_skill_relations_df,
            skills_df,
            st.session_state.occupation_skills_mapping,
            occupations_df,
            current_esco_role,
            current_manual_essential_skills,
            current_manual_optional_skills,
            current_removed_skills
        )
        
        if profile_for_skills:
            # Zeige aktuelle Skills mit Checkboxen zum Entfernen
            st.write("**Aktuelle Skills der Rolle:**")
            
            # Gruppiere Skills nach Typ - integriere manuelle Skills in die Hauptlisten
            essential_skills = [s for s in profile_for_skills['skills'] if s.get('is_essential', False)]
            optional_skills = [s for s in profile_for_skills['skills'] if not s.get('is_essential', False)]
            
            # Essential Skills
            if essential_skills:
                st.write("**Essential Skills:**")
                
                # Checkbox für "Alle Essential Skills auswählen"
                select_all_essential = st.checkbox(f"Alle Essential Skills auswählen ({len(essential_skills)} Skills)", key="select_all_essential")
                
                essential_to_remove = []
                essential_checkboxes = {}
                
                for i, skill in enumerate(essential_skills):
                    # Verwende die "Alle auswählen" Checkbox als Standardwert
                    default_value = select_all_essential
                    
                    # Individuelle Checkbox für jeden Skill
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_essential_{skill['skill_uri']}",
                        value=default_value
                    )
                    
                    if is_checked:
                        essential_to_remove.append(skill['skill_label'])
                    essential_checkboxes[skill['skill_label']] = is_checked
                
                # Anzeige der ausgewählten Essential Skills
                if essential_to_remove:
                    st.info(f"**{len(essential_to_remove)} von {len(essential_skills)} Essential Skills ausgewählt**")
                    st.warning(f"Folgende Essential Skills werden entfernt: {', '.join(essential_to_remove)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Essential Skills entfernen bestätigen", key="confirm_remove_essential"):
                            # Verarbeite die zu entfernenden Skills
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            current_manual_essential_list = [s.strip() for s in current_manual_essential_skills.split(';') if s.strip()]
                            
                            for skill_to_remove in essential_to_remove:
                                # Prüfe ob es ein manueller Essential Skill ist
                                if skill_to_remove in current_manual_essential_list:
                                    # Entferne aus manuellen Essential Skills
                                    current_manual_essential_list.remove(skill_to_remove)
                                else:
                                    # Füge zur Removed_Skills Liste hinzu (für automatische Skills)
                                    if skill_to_remove not in current_removed_list:
                                        current_removed_list.append(skill_to_remove)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            new_manual_essential_skills = '; '.join(current_manual_essential_list)
                            
                            # Aktualisiere die Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Essential_Skills'
                            ] = new_manual_essential_skills
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(essential_to_remove)} Essential Skills entfernt!")
                            else:
                                st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_essential_selection"):
                            st.rerun()
            
            # Optional Skills
            if optional_skills:
                st.write("**Optional Skills:**")
                
                # Checkbox für "Alle Optional Skills auswählen"
                select_all_optional = st.checkbox(f"Alle Optional Skills auswählen ({len(optional_skills)} Skills)", key="select_all_optional")
                
                optional_to_remove = []
                optional_checkboxes = {}
                
                for i, skill in enumerate(optional_skills):
                    # Verwende die "Alle auswählen" Checkbox als Standardwert
                    default_value = select_all_optional
                    
                    # Individuelle Checkbox für jeden Skill
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_optional_{skill['skill_uri']}",
                        value=default_value
                    )
                    
                    if is_checked:
                        optional_to_remove.append(skill['skill_label'])
                    optional_checkboxes[skill['skill_label']] = is_checked
                
                # Anzeige der ausgewählten Optional Skills
                if optional_to_remove:
                    st.info(f"**{len(optional_to_remove)} von {len(optional_skills)} Optional Skills ausgewählt**")
                    st.warning(f"Folgende Optional Skills werden entfernt: {', '.join(optional_to_remove)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Optional Skills entfernen bestätigen", key="confirm_remove_optional"):
                            # Verarbeite die zu entfernenden Skills
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            current_manual_optional_list = [s.strip() for s in current_manual_optional_skills.split(';') if s.strip()]
                            
                            for skill_to_remove in optional_to_remove:
                                # Prüfe ob es ein manueller Optional Skill ist
                                if skill_to_remove in current_manual_optional_list:
                                    # Entferne aus manuellen Optional Skills
                                    current_manual_optional_list.remove(skill_to_remove)
                                else:
                                    # Füge zur Removed_Skills Liste hinzu (für automatische Skills)
                                    if skill_to_remove not in current_removed_list:
                                        current_removed_list.append(skill_to_remove)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            new_manual_optional_skills = '; '.join(current_manual_optional_list)
                            
                            # Aktualisiere die Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Optional_Skills'
                            ] = new_manual_optional_skills
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(optional_to_remove)} Optional Skills entfernt!")
                            else:
                                st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_optional_selection"):
                            st.rerun()
            

            
            # Lade alle verfügbaren ESCO-Skills für Dropdown-Auswahl
            available_esco_skills = get_all_available_esco_skills(skills_df, skills_en_df)
            
            # Neue Essential Skills hinzufügen
            st.write("**Neue Essential Skills hinzufügen:**")
            
            # Erstelle Dropdown-Optionen für Essential Skills
            essential_skill_options = ["Bitte wählen Sie einen Essential Skill..."]
            essential_skill_labels = {}
            
            for skill in available_esco_skills:
                option_label = skill['display_label']
                essential_skill_options.append(option_label)
                essential_skill_labels[option_label] = skill['german_label']
            
            # Dropdown für neue Essential Skills
            selected_essential_skill = st.selectbox(
                "Wählen Sie einen Essential Skill aus:",
                essential_skill_options,
                key="essential_skill_dropdown"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Essential Skill hinzufügen", key="add_essential_skill"):
                    if selected_essential_skill and selected_essential_skill != "Bitte wählen Sie einen Essential Skill...":
                        # Füge den Skill zur Liste hinzu
                        skill_label = essential_skill_labels[selected_essential_skill]
                        current_essential_list = [s.strip() for s in current_manual_essential_skills.split(';') if s.strip()]
                        
                        if skill_label not in current_essential_list:
                            current_essential_list.append(skill_label)
                            new_essential_skills = '; '.join(current_essential_list)
                            
                            # Aktualisiere die Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Essential_Skills'
                            ] = new_essential_skills
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Essential Skill '{skill_label}' hinzugefügt!")
                            else:
                                st.warning("Skill hinzugefügt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.warning(f"Skill '{skill_label}' ist bereits hinzugefügt!")
                    else:
                        st.warning("Bitte wählen Sie einen Skill aus!")
            
            with col2:
                if st.button("Alle Essential Skills entfernen", key="remove_all_essential"):
                    st.session_state.employees_data.loc[
                        st.session_state.employees_data['Employee_ID'] == employee_id, 
                        'Manual_Essential_Skills'
                    ] = ''
                    
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success("Alle Essential Skills entfernt!")
                    else:
                        st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                    
                    st.rerun()
            
            st.markdown("---")
            
            # Neue Optional Skills hinzufügen
            st.write("**Neue Optional Skills hinzufügen:**")
            
            # Erstelle Dropdown-Optionen für Optional Skills
            optional_skill_options = ["Bitte wählen Sie einen Optional Skill..."]
            optional_skill_labels = {}
            
            for skill in available_esco_skills:
                option_label = skill['display_label']
                optional_skill_options.append(option_label)
                optional_skill_labels[option_label] = skill['german_label']
            
            # Dropdown für neue Optional Skills
            selected_optional_skill = st.selectbox(
                "Wählen Sie einen Optional Skill aus:",
                optional_skill_options,
                key="optional_skill_dropdown"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Optional Skill hinzufügen", key="add_optional_skill"):
                    if selected_optional_skill and selected_optional_skill != "Bitte wählen Sie einen Optional Skill...":
                        # Füge den Skill zur Liste hinzu
                        skill_label = optional_skill_labels[selected_optional_skill]
                        current_optional_list = [s.strip() for s in current_manual_optional_skills.split(';') if s.strip()]
                        
                        if skill_label not in current_optional_list:
                            current_optional_list.append(skill_label)
                            new_optional_skills = '; '.join(current_optional_list)
                            
                            # Aktualisiere die Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Manual_Optional_Skills'
                            ] = new_optional_skills
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Optional Skill '{skill_label}' hinzugefügt!")
                            else:
                                st.warning("Skill hinzugefügt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.warning(f"Skill '{skill_label}' ist bereits hinzugefügt!")
                    else:
                        st.warning("Bitte wählen Sie einen Skill aus!")
            
            with col2:
                if st.button("Alle Optional Skills entfernen", key="remove_all_optional"):
                    st.session_state.employees_data.loc[
                        st.session_state.employees_data['Employee_ID'] == employee_id, 
                        'Manual_Optional_Skills'
                    ] = ''
                    
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success("Alle Optional Skills entfernt!")
                    else:
                        st.warning("Skills entfernt, aber Speichern fehlgeschlagen!")
                    
                    st.rerun()
            
            # Entfernte Skills anzeigen
            if current_removed_skills and current_removed_skills.strip():
                st.write("**Aktuell entfernte Skills:**")
                removed_skills_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                
                # Checkbox für "Alle entfernten Skills auswählen"
                select_all_removed = st.checkbox(f"Alle entfernten Skills auswählen ({len(removed_skills_list)} Skills)", key="select_all_removed")
                
                removed_to_restore = []
                removed_checkboxes = {}
                
                for i, skill in enumerate(removed_skills_list):
                    # Verwende die "Alle auswählen" Checkbox als Standardwert
                    default_value = select_all_removed
                    
                    # Individuelle Checkbox für jeden entfernten Skill
                    is_checked = st.checkbox(
                        f"Wiederherstellen: {skill}", 
                        key=f"restore_removed_{skill.lower().replace(' ', '_')}",
                        value=default_value
                    )
                    
                    if is_checked:
                        removed_to_restore.append(skill)
                    removed_checkboxes[skill] = is_checked
                
                # Anzeige der ausgewählten entfernten Skills
                if removed_to_restore:
                    st.info(f"**{len(removed_to_restore)} von {len(removed_skills_list)} entfernten Skills ausgewählt**")
                    st.warning(f"Folgende Skills werden wiederhergestellt: {', '.join(removed_to_restore)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Entfernte Skills wiederherstellen bestätigen", key="confirm_restore_removed"):
                            # Entferne die ausgewählten Skills aus der Removed_Skills Liste
                            current_removed_list = [s.strip() for s in current_removed_skills.split(';') if s.strip()]
                            for skill_to_restore in removed_to_restore:
                                if skill_to_restore in current_removed_list:
                                    current_removed_list.remove(skill_to_restore)
                            
                            new_removed_skills = '; '.join(current_removed_list)
                            
                            # Aktualisiere die Session State Daten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Removed_Skills'
                            ] = new_removed_skills
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"{len(removed_to_restore)} entfernte Skills wiederhergestellt!")
                            else:
                                st.warning("Skills wiederhergestellt, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Auswahl zurücksetzen", key="reset_removed_selection"):
                            st.rerun()
                else:
                    # Button zum Wiederherstellen aller entfernten Skills (falls keine ausgewählt sind)
                    if st.button("Alle entfernten Skills wiederherstellen", key="restore_all_removed"):
                        st.session_state.employees_data.loc[
                            st.session_state.employees_data['Employee_ID'] == employee_id, 
                            'Removed_Skills'
                        ] = ''
                        
                        if save_employees_to_csv(st.session_state.employees_data):
                            st.success("Alle entfernten Skills wiederhergestellt!")
                        else:
                            st.warning("Skills wiederhergestellt, aber Speichern fehlgeschlagen!")
                        
                        st.rerun()
            else:
                st.write("**Entfernte Skills:** Keine Skills entfernt")
            
            # Allgemeine Zurücksetzen-Funktion
            st.markdown("---")
            st.write("**Alle Skill-Anpassungen zurücksetzen:**")
            if st.button("Alle Skill-Anpassungen zurücksetzen", key="reset_all_skills"):
                # Setze alle Skill-Anpassungen zurück
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                    ['Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
                ] = ['', '', '']
                
                # Speichere in CSV
                if save_employees_to_csv(st.session_state.employees_data):
                    st.success("Alle Skill-Anpassungen wurden zurückgesetzt!")
                else:
                    st.warning("Zurücksetzung gespeichert, aber CSV-Export fehlgeschlagen!")
                
                st.rerun()
        else:
            st.warning("Konnte kein Kompetenzprofil für Skill-Anpassungen erstellen.")
    


def show_occupation_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df):
    st.header("Berufsabgleich")
    
    # Verwende aktualisierte Mitarbeiterdaten aus Session State
    employees_df = st.session_state.employees_data
    
    if employees_df.empty:
        st.warning("Keine Mitarbeiterdaten gefunden.")
        return
    
    # Verwende die globale Mitarbeiterauswahl
    if 'current_employee_id' not in st.session_state:
        st.info("Bitte wählen Sie einen Mitarbeiter in der Sidebar aus.")
        return
    
    employee_id = st.session_state.current_employee_id
    
    # Hole aktuelle Daten aus Session State (nicht gecacht)
    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
    
    st.subheader(f"Berufsabgleich für {current_employee_data.get('Name', employee_id)}")
    
    # Aktuelle Rolle des Mitarbeiters
    current_kldb = current_employee_data.get('KldB_5_digit', '')
    current_manual_skills = current_employee_data.get('Manual_Skills', '')
    current_esco_role = current_employee_data.get('ESCO_Role', '')
    
    # Hole zusätzliche Skill-Daten aus Session State
    current_manual_essential_skills = current_employee_data.get('Manual_Essential_Skills', '')
    current_manual_optional_skills = current_employee_data.get('Manual_Optional_Skills', '')
    current_removed_skills = current_employee_data.get('Removed_Skills', '')
    
    # Erstelle das aktuelle Mitarbeiterprofil
    current_profile = create_employee_profile(
        employee_id,
        current_kldb,
        current_manual_skills,
        kldb_esco_df,
        occupation_skill_relations_df,
        skills_df,
        st.session_state.occupation_skills_mapping,
        occupations_df,
        current_esco_role,  # Übergebe die gespeicherte ESCO-Rolle
        current_manual_essential_skills,
        current_manual_optional_skills,
        current_removed_skills
    )
    
    if not current_profile:
        st.error("Konnte kein aktuelles Mitarbeiterprofil erstellen.")
        return
    
    # Zeige aktuelle Rolle und ausgewählte Zielrolle nebeneinander
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Rolle:**")
        current_role = current_profile['current_role']
        
        # Zeige die gespeicherte ESCO-Rolle oder die berechnete
        if current_esco_role:
            st.write(f"• KldB: {current_role.get('KldB_Label', 'N/A')} ({current_role.get('KldB_Code', 'N/A')})")
            st.write(f"• ESCO: {current_esco_role}")
            st.write(f"• Anzahl Skills: {len(current_profile['skills'])}")
        else:
            st.write(f"• KldB: {current_role.get('KldB_Label', 'N/A')} ({current_role.get('KldB_Code', 'N/A')})")
            st.write(f"• ESCO: {current_role.get('ESCO_Label', 'N/A')}")
            st.write(f"• Anzahl Skills: {len(current_profile['skills'])}")
    
    with col2:
        st.write("**Ausgewählte Zielrolle:**")
        # Zeige die ausgewählte Zielrolle basierend auf Session State oder aktueller Dropdown-Auswahl
        if 'selected_target_role' in st.session_state:
            target_role = st.session_state.selected_target_role
            st.write(f"• KldB: {target_role['KldB_Label']} ({target_role['KldB_Code']})")
            st.write(f"• ESCO: {target_role['ESCO_Label']} ({target_role['ESCO_Code']})")
            
            # Button zum Zurücksetzen der Zielrolle
            if st.button("Zielrolle zurücksetzen", key="reset_target_role_matching", type="secondary"):
                # Lösche die gespeicherte Zielrolle aus den Mitarbeiterdaten
                st.session_state.employees_data.loc[
                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                    ['Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label']
                ] = ['', '', '', '']
                
                # Speichere in CSV
                if save_employees_to_csv(st.session_state.employees_data):
                    # Lösche aus Session State
                    if 'selected_target_role' in st.session_state:
                        del st.session_state.selected_target_role
                    st.success("Zielrolle wurde zurückgesetzt und gespeichert!")
                else:
                    st.warning("Zielrolle zurückgesetzt, aber Speichern fehlgeschlagen!")
                
                st.rerun()
        else:
            st.write("• Keine Zielrolle ausgewählt")
    
    st.markdown("---")
    
    # Zielrolle auswählen
    st.subheader("Neue Zielrolle auswählen")
    
    # Erstelle eine Dropdown-Box mit allen verfügbaren KldB-Rollen
    # Entferne Duplikate basierend auf KldB_Code UND KldB_Label
    available_kldb_roles = kldb_esco_df[['KldB_Code', 'KldB_Label']].drop_duplicates(subset=['KldB_Code', 'KldB_Label'])
    available_kldb_roles = available_kldb_roles.sort_values('KldB_Label')
    
    # Erstelle Optionen für die Dropdown-Box - kürzere, saubere Anzeige
    kldb_options = []
    seen_options = set()  # Verhindert Duplikate in der Anzeige
    
    for _, row in available_kldb_roles.iterrows():
        kldb_label = str(row['KldB_Label']).strip()
        kldb_code = str(row['KldB_Code']).strip()
        
        # Überspringe leere Einträge
        if not kldb_label or not kldb_code or kldb_label == 'nan' or kldb_code == 'nan':
            continue
        
        # Kürze lange Labels für bessere Lesbarkeit
        display_label = kldb_label
        if len(display_label) > 40:
            display_label = display_label[:37] + "..."
        
        # Erstelle saubere Option
        option = f"{display_label} | {kldb_code}"
        
        # Verhindere Duplikate in der Anzeige
        if option not in seen_options:
            kldb_options.append(option)
            seen_options.add(option)
    
    # Sortiere die Optionen alphabetisch
    kldb_options.sort()
    kldb_options.insert(0, "Bitte wählen Sie eine neue Zielrolle...")
    
    selected_target_role = st.selectbox("KldB-Zielrolle auswählen:", kldb_options, help="Scrollen Sie durch die Liste oder tippen Sie den Anfang des Berufsnamens")
    
    # Zeige dynamische Übersicht basierend auf aktueller Auswahl
    if selected_target_role and selected_target_role != "Bitte wählen Sie eine neue Zielrolle...":
        # Extrahiere KldB-Code aus der Auswahl
        kldb_code = selected_target_role.split(" | ")[1]
        kldb_label = selected_target_role.split(" | ")[0]
        
        # Finde das vollständige Label aus den Originaldaten
        full_label = available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ]['KldB_Label'].iloc[0] if not available_kldb_roles[
            (available_kldb_roles['KldB_Code'] == kldb_code) & 
            (available_kldb_roles['KldB_Label'].str.contains(kldb_label.split('...')[0] if '...' in kldb_label else kldb_label, na=False))
        ].empty else kldb_label
        
        # Zeige dynamische Übersicht der ausgewählten KldB-Rolle
        st.markdown("---")
        st.write("**Aktuell ausgewählte KldB-Zielrolle:**")
        st.write(f"• KldB: {full_label} ({kldb_code})")
        
        # Finde alle ESCO-Rollen für die ausgewählte KldB-Rolle
        target_roles = kldb_esco_df[kldb_esco_df['KldB_Code'] == kldb_code]
        
        if not target_roles.empty:
            st.write(f"**Verfügbare ESCO-Rollen für Zielrolle '{full_label}':**")
            
            # Zeige alle verfügbaren ESCO-Rollen mit Expander
            for idx, role in target_roles.iterrows():
                esco_label = role['ESCO_Label']
                esco_code = role['ESCO_Code']
                
                # Hole Skills für diese ESCO-Rolle
                role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                
                with st.expander(f"{esco_label} ({esco_code})"):
                    if role_skills:
                        # Legende für Skill-Farbpunkte
                        st.markdown("**Skill-Legende:**")
                        legend_col1, legend_col2 = st.columns(2)
                        with legend_col1:
                            st.write("**Essential Skills** - Unverzichtbare Skills")
                        with legend_col2:
                            st.write("**Optional Skills** - Hilfreiche Skills")
                        
                        st.markdown("---")
                        
                        st.write("**Skills:**")
                        for skill in role_skills:
                            essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                            st.write(f"• {skill['skill_label']}{essential_mark}")
                        
                        # Button zum Auswählen als Zielrolle für den Vergleich
                        if st.button(f"Als Zielrolle für Vergleich auswählen", key=f"select_target_role_{idx}"):
                            # Prüfe ob es die gleiche ESCO-Rolle ist
                            current_esco_role = current_employee_data.get('ESCO_Role', '')
                            
                            if current_esco_role and esco_label == current_esco_role:
                                st.warning("Sie haben die gleiche ESCO-Rolle wie die aktuelle Rolle ausgewählt. Bitte wählen Sie eine andere Zielrolle.")
                            else:
                                # Erstelle ein Zielrollen-Profil für den Vergleich
                                target_role_data = {
                                    'ESCO_Label': esco_label,
                                    'ESCO_Code': esco_code,
                                    'KldB_Label': full_label,
                                    'KldB_Code': kldb_code
                                }
                                # Speichere die ausgewählte Zielrolle im Session State für andere Sektionen
                                st.session_state.selected_target_role = target_role_data
                                # Speichere die Zielrolle persistent für den aktuellen Mitarbeiter
                                employees_df.loc[employees_df['Employee_ID'] == employee_id, 'Target_KldB_Code'] = kldb_code
                                employees_df.loc[employees_df['Employee_ID'] == employee_id, 'Target_KldB_Label'] = full_label
                                employees_df.loc[employees_df['Employee_ID'] == employee_id, 'Target_ESCO_Code'] = esco_code
                                employees_df.loc[employees_df['Employee_ID'] == employee_id, 'Target_ESCO_Label'] = esco_label
                                save_employees_to_csv(employees_df)
                                st.session_state.employees_data = employees_df
                                # Zeige sofort die aktualisierte Übersicht
                                st.markdown("---")
                                st.success("Zielrolle ausgewählt!")
                                st.write("**Ausgewählte Zielrolle:**")
                                st.write(f"• KldB: {target_role_data['KldB_Label']} ({target_role_data['KldB_Code']})")
                                st.write(f"• ESCO: {target_role_data['ESCO_Label']} ({target_role_data['ESCO_Code']})")
                                st.markdown("---")
                                
                                # Berechne den Match zwischen aktueller und neuer Rolle
                                match_result = calculate_occupation_match(
                                    current_profile, 
                                    target_role_data, 
                                    occupation_skill_relations_df, 
                                    skills_df, 
                                    st.session_state.occupation_skills_mapping, 
                                    occupations_df
                                )
                                
                                if match_result:
                                    # Zeige Vergleich zwischen aktueller und neuer Rolle
                                    st.markdown("---")
                                    st.subheader("Rollenvergleich")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Aktuelle Rolle:**")
                                        st.write(f"• {current_role.get('KldB_Label', 'N/A')}")
                                        st.write(f"• {current_role.get('ESCO_Label', 'N/A')}")
                                        st.write(f"• Skills: {len(current_profile['skills'])}")
                                    
                                    with col2:
                                        st.write("**Neue Zielrolle:**")
                                        st.write(f"• {target_role_data['KldB_Label']}")
                                        st.write(f"• {target_role_data['ESCO_Label']}")
                                        st.write(f"• Skills: {len(match_result.get('matching_skills', []) + match_result.get('missing_skills', []))}")
                                    
                                    st.markdown("---")
                                    
                                    # Match-Ergebnisse
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if match_result['has_target_skills']:
                                            st.metric("Fit-Score % (nicht gewichtet)", f"{match_result['match_percentage']:.1f}%", help="Sie misst, wie gut das Kompetenzprofil eines Mitarbeiters zu den Anforderungen einer Zielrolle passt.")
                                        else:
                                            st.metric("Fit-Score % (nicht gewichtet)", "N/A", help="Sie misst, wie gut das Kompetenzprofil eines Mitarbeiters zu den Anforderungen einer Zielrolle passt.")
                                    
                                    with col2:
                                        if match_result['has_target_skills']:
                                            st.metric("Fit-Score % (gewichtet)", f"{match_result['weighted_fit_percentage']:.1f}%", help="Sie misst, wie gut ein Mitarbeiter für eine Zielrolle geeignet ist, wobei essenzielle (wichtige) Kompetenzen stärker gewichtet werden als optionale.")
                                        else:
                                            st.metric("Fit-Score % (gewichtet)", "N/A", help="Sie misst, wie gut ein Mitarbeiter für eine Zielrolle geeignet ist, wobei essenzielle (wichtige) Kompetenzen stärker gewichtet werden als optionale.")
                                    
                                    st.markdown("---")
                                    
                                    # Fehlende und vorhandene Skills
                                    col1, col2 = st.columns(2)
                                    
                                    # Legende für Skill-Farbpunkte
                                    st.markdown("**Skill-Legende:**")
                                    legend_col1, legend_col2 = st.columns(2)
                                    with legend_col1:
                                        st.write("**Essential Skills** - Unverzichtbare Skills (zählen doppelt)")
                                    with legend_col2:
                                        st.write("**Optional Skills** - Hilfreiche Skills (zählen einfach)")
                                    
                                    st.markdown("---")
                                    
                                    with col1:
                                        if match_result['missing_skills']:
                                            st.write("**Fehlende Skills für neue Rolle:**")
                                            for skill in match_result['missing_skills']:
                                                essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                                                st.write(f"• {skill['skill_label']}{essential_mark}")
                                        else:
                                            st.write("**Fehlende Skills:**")
                                            st.write("Alle benötigten Skills sind vorhanden!")
                                    
                                    with col2:
                                        if match_result['matching_skills']:
                                            st.write("**Bereits vorhandene Skills:**")
                                            for skill in match_result['matching_skills']:
                                                essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                                                st.write(f"• {skill['skill_label']}{essential_mark}")
                                        else:
                                            st.write("**Bereits vorhandene Skills:**")
                                            st.write("Keine Übereinstimmungen gefunden.")
                                    
                                    # Speichere Match-Ergebnis für Kursempfehlungen
                                    st.session_state.current_match = match_result
                                else:
                                    st.error("Fehler beim Berechnen des Matches.")
                    else:
                        st.write("Keine Skills für diese Rolle gefunden.")
        else:
            st.warning(f"Keine ESCO-Rollen für Zielrolle '{full_label}' gefunden.")
    else:
        st.info("Bitte wählen Sie eine neue Zielrolle aus der Dropdown-Box.")

def show_course_recommendations(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df):
    st.header("Kursempfehlungen")
    
    # Session State für Mitarbeiterdaten initialisieren, falls nicht vorhanden
    if 'employees_data' not in st.session_state:
        # Füge Name-Spalte hinzu, falls sie nicht existiert
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role'])
    
    # Verwende aktualisierte Mitarbeiterdaten aus Session State
    employees_df = st.session_state.employees_data
    
    # Prüfe ob Match-Ergebnis vorhanden
    if 'current_match' not in st.session_state:
        st.info("**Bitte führe zuerst einen Berufsabgleich durch.**")
        st.info("Gehe zu 'Berufsabgleich' und wähle einen Mitarbeiter und eine Zielrolle aus.")
        return
    
    match_result = st.session_state.current_match
    
    if not match_result['has_target_skills']:
        st.warning("Keine Skills für den Zielberuf verfügbar. Kursempfehlungen können nicht generiert werden.")
        st.info("Tipp: Wähle einen anderen Zielberuf oder überprüfe die ESCO-Daten.")
        return
    
    if not match_result['missing_skills']:
        st.success("**Alle benötigten Skills sind bereits vorhanden!**")
        return
    
    # Zeige Kontext-Informationen
    st.subheader("Kontext der Kursempfehlungen")
    
    # Hole Mitarbeiterinformationen aus dem Match-Ergebnis
    current_role = match_result.get('current_role', {})
    target_role = match_result.get('target_role', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mitarbeiter:**")
        st.write(f"• **Aktuelle Rolle:** {current_role.get('KldB_Label', 'N/A')}")
        st.write(f"• **Aktuelle ESCO-Rolle:** {current_role.get('ESCO_Label', 'N/A')}")
        st.write(f"• **Aktuelle Skills:** {len(match_result.get('matching_skills', []))}")
    
    with col2:
        st.write("**Zielrolle:**")
        st.write(f"• **Neue Rolle:** {target_role.get('KldB_Label', 'N/A')}")
        st.write(f"• **Neue ESCO-Rolle:** {target_role.get('ESCO_Label', 'N/A')}")
        st.write(f"• **Benötigte Skills:** {len(match_result.get('matching_skills', []) + match_result.get('missing_skills', []))}")
    
    st.markdown("---")
    
    # Zeige fehlende Skills
    st.subheader("Fehlende Skills")
    
    missing_skills = match_result['missing_skills']
    st.write(f"**Anzahl fehlender Skills:** {len(missing_skills)}")
    
    # Gruppiere nach Essential/Optional
    essential_missing = [skill for skill in missing_skills if skill.get('is_essential', False)]
    optional_missing = [skill for skill in missing_skills if not skill.get('is_essential', False)]
    
    if essential_missing:
        st.write("**Essential Skills (höchste Priorität):**")
        for skill in essential_missing:
            st.write(f"• {skill['skill_label']}")
    
    if optional_missing:
        st.write("**Optional Skills:**")
        for skill in optional_missing:
            st.write(f"• {skill['skill_label']}")
    
    st.markdown("---")
    
    # Extrahiere erweiterte Skills aus den missing_skills Dictionaries
    missing_skills_enhanced = []
    for skill in match_result['missing_skills']:
        # Erweitere Skills um englische Labels, falls verfügbar
        enhanced_skill = skill.copy()
        skill_uri = skill.get('skill_uri', '')
        
        if skill_uri in st.session_state.skill_mapping_with_english:
            english_label = st.session_state.skill_mapping_with_english[skill_uri]['english']
            enhanced_skill['skill_label_english'] = english_label
            enhanced_skill['skill_labels_combined'] = f"{skill['skill_label']} | {english_label}"
        else:
            enhanced_skill['skill_label_english'] = skill['skill_label']
            enhanced_skill['skill_labels_combined'] = skill['skill_label']
        
        missing_skills_enhanced.append(enhanced_skill)
    
    # Verwende die erweiterten Skills für Kursempfehlungen
    missing_skill_labels = missing_skills_enhanced
    
    # Finde Kursempfehlungen
    with st.spinner("Suche passende Kurse..."):
        recommendations = find_udemy_courses_for_skills(
            missing_skill_labels,
            udemy_courses_df,
            top_k=3
        )
    
    if recommendations:
        st.subheader(f"Top-Kursempfehlungen für fehlende Skills")
        
        # Legende für Skill-Farbpunkte
        st.markdown("**Skill-Legende:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Essential Skills** - Unverzichtbare Skills (höchste Priorität)")
        with col2:
            st.write("**Optional Skills** - Hilfreiche Skills (niedrigere Priorität)")
        
        st.markdown("---")
        
        # Gruppiere nach Skill
        skill_groups = {}
        for rec in recommendations:
            skill = rec['skill']
            if skill not in skill_groups:
                skill_groups[skill] = []
            skill_groups[skill].append(rec)
        
        for skill, courses in skill_groups.items():
            # Finde ob es ein Essential oder Optional Skill ist
            skill_type = "Essential" if any(s['skill_label'] == skill and s.get('is_essential', False) for s in missing_skills) else "Optional"
            
            st.write(f"**Für Skill: {skill}** ({skill_type})")
            
            for i, course in enumerate(courses[:3], 1):  # Top 3 Kurse pro Skill
                with st.expander(f"{i}. {course['course_title']} (Score: {course['similarity_score']:.3f})"):
                    st.write(f"**Headline:** {course['course_headline']}")
                    st.write(f"**Beschreibung:** {course['course_description']}")
                    st.write(f"**Preis:** {course['course_price']}")
                    st.write(f"**Sprache:** {course['course_language']}")
                    st.markdown(f"[Zum Kurs auf Udemy]({course['course_url']})")
            
            st.markdown("---")
    else:
        st.warning("Keine passenden Kurse gefunden.")
        st.info("Tipp: Überprüfe die Udemy-Kursdaten oder versuche es mit einem anderen Zielberuf.")
    
    # Debugging-Sektion: Zeige alle Skills und ihre gefundenen Kurse (auch unter der Schwelle)
    st.markdown("---")
    st.subheader("Debugging: Alle gefundenen Kurse pro Skill")
    st.info("Diese Sektion zeigt alle Kurse, die für jeden Skill gefunden wurden, auch wenn sie unter der Ähnlichkeits-Schwelle von 0.01 liegen.")
    
    # Erstelle eine erweiterte Debugging-Funktion
    def find_all_courses_for_skill_debug(skill, udemy_courses_df, top_k=10):
        """Findet alle Kurse für einen Skill mit Debugging-Informationen"""
        if udemy_courses_df.empty:
            return []
        
        # Bereite Kursdaten vor (falls noch nicht geschehen)
        if 'processed_text' not in udemy_courses_df.columns:
            udemy_courses_df['processed_text'] = (
                udemy_courses_df['Title'].fillna('') + ' ' +
                udemy_courses_df['Headline'].fillna('') + ' ' +
                udemy_courses_df['Description'].fillna('')
            ).apply(preprocess_text)
        
        # TF-IDF Vektorisierung
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        course_vectors = vectorizer.fit_transform(udemy_courses_df['processed_text'])
        
        # Bereite Skill-Text vor
        skill_text = preprocess_text(skill)
        skill_vector = vectorizer.transform([skill_text])
        
        # Berechne Ähnlichkeiten
        similarities = cosine_similarity(skill_vector, course_vectors).flatten()
        
        # Top-K Kurse (auch unter der Schwelle)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        debug_recommendations = []
        for idx in top_indices:
            course = udemy_courses_df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Bestimme Status basierend auf Schwelle
            status = "Empfohlen" if similarity_score > 0.01 else "Unter Schwelle"
            
            debug_recommendations.append({
                'skill': skill,
                'course_title': course['Title'],
                'course_headline': course['Headline'],
                'course_description': course['Description'][:200] + '...' if len(str(course['Description'])) > 200 else course['Description'],
                'course_url': course['URL'],
                'course_price': course['Price'],
                'course_language': course['Language'],
                'similarity_score': similarity_score,
                'status': status
            })
        
        return debug_recommendations
    
    # Zeige Debugging-Informationen für jeden fehlenden Skill
    for skill in missing_skill_labels:
        # Bestimme Skill-Typ basierend auf dem Skill-Objekt
        if isinstance(skill, dict):
            skill_name = skill.get('skill_label', str(skill))
            skill_type = "Essential" if skill.get('is_essential', False) else "Optional"
        else:
            skill_name = str(skill)
            skill_type = "Essential" if any(s['skill_label'] == skill and s.get('is_essential', False) for s in missing_skills) else "Optional"
        
        with st.expander(f"Debug: {skill_name} ({skill_type})"):
            # Verwende das kombinierte Label für die Kurssuche
            if isinstance(skill, dict):
                search_skill = skill.get('skill_labels_combined', skill.get('skill_label', str(skill)))
            else:
                search_skill = str(skill)
            
            debug_courses = find_all_courses_for_skill_debug(search_skill, udemy_courses_df, top_k=10)
            
            if debug_courses:
                st.write(f"**Gefundene Kurse für '{skill_name}':**")
                
                # Zeige Statistiken
                recommended_count = sum(1 for course in debug_courses if course['status'] == "Empfohlen")
                total_count = len(debug_courses)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Empfohlene Kurse", recommended_count)
                with col2:
                    st.metric("Gefundene Kurse (gesamt)", total_count)
                
                st.markdown("---")
                
                # Zeige alle Kurse mit Status
                for i, course in enumerate(debug_courses, 1):
                    status_color = "Empfohlen" if course['status'] == "Empfohlen" else "Unter Schwelle"
                    st.write(f"**{i}. {status_color} {course['course_title']} (Score: {course['similarity_score']:.4f})**")
                    st.write(f"**Status:** {course['status']}")
                    st.write(f"**Ähnlichkeits-Score:** {course['similarity_score']:.4f}")
                    st.write(f"**Schwelle:** 0.01")
                    st.write(f"**Headline:** {course['course_headline']}")
                    st.write(f"**Beschreibung:** {course['course_description']}")
                    st.write(f"**Preis:** {course['course_price']}")
                    st.write(f"**Sprache:** {course['course_language']}")
                    st.markdown(f"[Zum Kurs auf Udemy]({course['course_url']})")
                    st.markdown("---")
            else:
                st.warning(f"Keine Kurse für Skill '{skill_name}' gefunden.")
                st.info("Mögliche Gründe:")
                st.write("• Skill-Text konnte nicht verarbeitet werden")
                st.write("• Keine passenden Kurse in der Datenbank")
                st.write("• TF-IDF Vektorisierung fehlgeschlagen")

def show_overview(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df):
    st.header("Gesamtübersicht")
    
    # Session State für Mitarbeiterdaten initialisieren, falls nicht vorhanden
    if 'employees_data' not in st.session_state:
        # Füge Name-Spalte hinzu, falls sie nicht existiert
        if 'Name' not in employees_df.columns:
            employees_df['Name'] = 'Unbekannt'
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role'])
    
    # Verwende aktualisierte Mitarbeiterdaten aus Session State
    employees_df = st.session_state.employees_data
    
    # Zeige aktuelle Mitarbeiter- und Zielrollen-Informationen
    if 'current_employee_id' in st.session_state and not employees_df.empty:
        employee_id = st.session_state.current_employee_id
        employee_data = employees_df[employees_df['Employee_ID'] == employee_id]
        
        if not employee_data.empty:
            employee_data = employee_data.iloc[0]
            st.subheader(f"Aktueller Mitarbeiter: {employee_data.get('Name', employee_id)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Aktuelle Rolle:**")
                current_kldb = employee_data.get('KldB_5_digit', '')
                current_esco_role = employee_data.get('ESCO_Role', '')
                
                if current_esco_role:
                    st.write(f"• KldB-Code: {current_kldb}")
                    st.write(f"• ESCO-Rolle: {current_esco_role}")
                    
                    # Finde die KldB-Rolle basierend auf dem Code
                    kldb_match = kldb_esco_df[kldb_esco_df['KldB_Code'] == current_kldb]
                    if not kldb_match.empty:
                        st.write(f"• KldB-Rolle: {kldb_match.iloc[0]['KldB_Label']}")
                else:
                    st.write(f"• KldB-Code: {current_kldb}")
                    st.write("• ESCO-Rolle: Nicht zugewiesen")
            
            with col2:
                st.write("**Ausgewählte Zielrolle:**")
                if 'selected_target_role' in st.session_state:
                    target_role = st.session_state.selected_target_role
                    st.write(f"• KldB-Code: {target_role['KldB_Code']}")
                    st.write(f"• KldB-Rolle: {target_role['KldB_Label']}")
                    st.write(f"• ESCO-Rolle: {target_role['ESCO_Label']}")
                    st.write(f"• ESCO-Code: {target_role['ESCO_Code']}")
                else:
                    st.write("• Keine Zielrolle ausgewählt")
            
            st.markdown("---")
    
    # Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mitarbeiter", len(employees_df))
    
    with col2:
        st.metric("ESCO-Berufe", len(get_all_esco_occupations(kldb_esco_df)))
    
    with col3:
        st.metric("KldB-ESCO Mappings", len(kldb_esco_df))
    
    with col4:
        st.metric("Udemy-Kurse", len(udemy_courses_df) if not udemy_courses_df.empty else 0)
    
    # Export-Funktion
    st.subheader("Export")
    
    if st.button("Exportiere alle Ergebnisse als CSV"):
        # Hier könnte die Export-Logik implementiert werden
        st.success("Export-Funktion wird implementiert...")
    
    # Datenqualität
    st.subheader("Datenqualität")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**KldB-ESCO Mapping:**")
        st.write(f"• Eindeutige KldB-Codes: {kldb_esco_df['KldB_Code'].nunique()}")
        st.write(f"• Eindeutige ESCO-Codes: {kldb_esco_df['ESCO_Code'].nunique()}")
        st.write(f"• ESCO Beruf-Skill Beziehungen: {len(occupation_skill_relations_df)}")
    
    with col2:
        st.write("**Udemy-Kurse:**")
        if not udemy_courses_df.empty:
            st.write(f"• Kurse mit Preis: {udemy_courses_df['Price'].notna().sum()}")
            st.write(f"• Sprachen: {udemy_courses_df['Language'].nunique()}")
        else:
            st.write("• Keine Udemy-Daten verfügbar")

def show_employee_management(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df):
    st.header("Mitarbeiter-Verwaltung")
    
    # Session State für Mitarbeiterdaten initialisieren
    if 'employees_data' not in st.session_state:
        # Lade Mitarbeiterdaten aus CSV oder verwende Standarddaten
        csv_employees = load_employees_from_csv()
        if not csv_employees.empty:
            st.session_state.employees_data = csv_employees
            st.success("Mitarbeiterdaten aus CSV geladen")
        else:
            # Füge Name-Spalte hinzu, falls sie nicht existiert
            if 'Name' not in employees_df.columns:
                employees_df['Name'] = 'Unbekannt'
            st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
    # Sidebar für Navigation
    st.sidebar.subheader("Verwaltungsoptionen")
    management_option = st.sidebar.selectbox(
        "Wählen Sie eine Aktion:",
        ["Mitarbeiter anzeigen", "Neuen Mitarbeiter anlegen", "Mitarbeiter bearbeiten", "Mitarbeiter löschen"]
    )
    
    if management_option == "Mitarbeiter anzeigen":
        st.subheader("Alle Mitarbeiter")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter vorhanden. Legen Sie einen neuen Mitarbeiter an.")
        else:
            # Zeige alle Mitarbeiter in einer Tabelle
            display_columns = ['Employee_ID', 'Name', 'KldB_5_digit', 'ESCO_Role', 'Manual_Skills', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills']
            available_columns = [col for col in display_columns if col in st.session_state.employees_data.columns]
            
            st.dataframe(
                st.session_state.employees_data[available_columns],
                use_container_width=True
            )
            
            # Download-Funktion
            csv = st.session_state.employees_data.to_csv(index=False)
            st.download_button(
                label="Mitarbeiterdaten als CSV herunterladen",
                data=csv,
                file_name="mitarbeiter_daten.csv",
                mime="text/csv"
            )
    
    elif management_option == "Neuen Mitarbeiter anlegen":
        st.subheader("Neuen Mitarbeiter anlegen")
        
        with st.form("new_employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("Vorname *", key="new_first_name")
            
            with col2:
                last_name = st.text_input("Nachname *", key="new_last_name")
            
            # Generiere automatisch eine Employee_ID
            if first_name and last_name:
                # Erstelle eine eindeutige ID basierend auf Name und Zeitstempel
                timestamp = int(time.time())
                employee_id = f"{first_name[:3].upper()}{last_name[:3].upper()}{timestamp % 10000:04d}"
                st.info(f"**Generierte Employee-ID:** {employee_id}")
            
            submitted = st.form_submit_button("Mitarbeiter anlegen")
            
            if submitted:
                if first_name and last_name:
                    # Erstelle neuen Mitarbeiter
                    new_employee = {
                        'Employee_ID': employee_id,
                        'Name': f"{first_name} {last_name}",
                        'KldB_5_digit': '',
                        'Manual_Skills': '',
                        'ESCO_Role': '',
                        'Target_KldB_Code': '',
                        'Target_KldB_Label': '',
                        'Target_ESCO_Code': '',
                        'Target_ESCO_Label': '',
                        'Manual_Essential_Skills': '',
                        'Manual_Optional_Skills': '',
                        'Removed_Skills': ''
                    }
                    
                    # Füge zur Session State hinzu
                    st.session_state.employees_data = pd.concat([
                        st.session_state.employees_data,
                        pd.DataFrame([new_employee])
                    ], ignore_index=True)
                    
                    # Speichere in CSV
                    if save_employees_to_csv(st.session_state.employees_data):
                        st.success(f"Mitarbeiter '{first_name} {last_name}' erfolgreich angelegt und gespeichert!")
                    else:
                        st.warning(f"Mitarbeiter angelegt, aber Speichern fehlgeschlagen!")
                    
                    # Formular zurücksetzen
                    st.rerun()
                else:
                    st.error("Vorname und Nachname sind Pflichtfelder!")
    
    elif management_option == "Mitarbeiter bearbeiten":
        st.subheader("Mitarbeiter bearbeiten")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter zum Bearbeiten vorhanden.")
        else:
            # Wähle Mitarbeiter aus
            employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in st.session_state.employees_data.iterrows()]
            selected_employee_str = st.selectbox("Mitarbeiter auswählen:", employee_options)
            
            if selected_employee_str:
                employee_id = selected_employee_str.split(" - ")[0]
                employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
                
                with st.form("edit_employee_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Extrahiere Vor- und Nachname
                        full_name = employee_data.get('Name', 'Unbekannt')
                        name_parts = full_name.split(' ', 1)
                        first_name = name_parts[0] if len(name_parts) > 0 else ""
                        last_name = name_parts[1] if len(name_parts) > 1 else ""
                        
                        new_first_name = st.text_input("Vorname *", value=first_name, key="edit_first_name")
                    
                    with col2:
                        new_last_name = st.text_input("Nachname *", value=last_name, key="edit_last_name")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submitted = st.form_submit_button("Änderungen speichern")
                    with col2:
                        if st.form_submit_button("Abbrechen"):
                            st.rerun()
                    
                    if submitted:
                        if new_first_name and new_last_name:
                            # Aktualisiere Mitarbeiterdaten
                            st.session_state.employees_data.loc[
                                st.session_state.employees_data['Employee_ID'] == employee_id, 
                                'Name'
                            ] = f"{new_first_name} {new_last_name}"
                            
                            # Speichere in CSV
                            if save_employees_to_csv(st.session_state.employees_data):
                                st.success(f"Mitarbeiter '{new_first_name} {new_last_name}' erfolgreich aktualisiert und gespeichert!")
                            else:
                                st.warning(f"Mitarbeiter aktualisiert, aber Speichern fehlgeschlagen!")
                            
                            st.rerun()
                        else:
                            st.error("Vorname und Nachname sind Pflichtfelder!")
    
    elif management_option == "Mitarbeiter löschen":
        st.subheader("Mitarbeiter löschen")
        
        if st.session_state.employees_data.empty:
            st.info("Keine Mitarbeiter zum Löschen vorhanden.")
        else:
            # Wähle Mitarbeiter aus
            employee_options = [f"{row['Employee_ID']} - {row.get('Name', 'Unbekannt')}" for _, row in st.session_state.employees_data.iterrows()]
            selected_employee_str = st.selectbox("Mitarbeiter zum Löschen auswählen:", employee_options, key="delete_employee")
            
            if selected_employee_str:
                employee_id = selected_employee_str.split(" - ")[0]
                employee_name = selected_employee_str.split(" - ")[1]
                
                st.warning(f"Sie sind dabei, den Mitarbeiter '{employee_name}' zu löschen!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Endgültig löschen", type="primary"):
                        # Lösche Mitarbeiter
                        st.session_state.employees_data = st.session_state.employees_data[
                            st.session_state.employees_data['Employee_ID'] != employee_id
                        ]
                        
                        # Speichere in CSV
                        if save_employees_to_csv(st.session_state.employees_data):
                            st.success(f"Mitarbeiter '{employee_name}' erfolgreich gelöscht und Änderungen gespeichert!")
                        else:
                            st.warning(f"Mitarbeiter gelöscht, aber Speichern fehlgeschlagen!")
                        
                        st.rerun()
                
                with col2:
                    if st.button("Abbrechen"):
                        st.rerun()
    
    # Aktualisiere die globale employees_df für andere Sektionen
    if 'employees_data' in st.session_state:
        globals()['employees_df'] = st.session_state.employees_data

if __name__ == "__main__":
    main()