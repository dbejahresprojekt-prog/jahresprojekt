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
import xml.etree.ElementTree as ET
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
        
        # Lade XML-Daten aus Archi
        archi_xml_path = 'data/DigiVan.xml'
        archi_data = None
        if os.path.exists(archi_xml_path):
            st.write(f"📁 XML-Datei gefunden: {archi_xml_path}")
            st.write(f"📊 Dateigröße: {os.path.getsize(archi_xml_path)} Bytes")
            
            try:
                archi_data = parse_archi_xml(archi_xml_path)
                if archi_data:
                    st.success(f"✅ Archi XML-Daten erfolgreich geladen: {len(archi_data.get('capabilities', []))} Capabilities, {len(archi_data.get('resources', []))} Resources")
                else:
                    st.warning("⚠️ Archi XML-Daten konnten nicht geparst werden")
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der XML-Daten: {str(e)}")
                archi_data = None
        else:
            st.warning(f"⚠️ Archi XML-Datei nicht gefunden: {archi_xml_path}")
            st.write("**Erwarteter Pfad:** data/DigiVan.xml")
        
        # Lade Kompetenzabgleich XML-Daten
        kompetenzabgleich_data = None
        kompetenzabgleich_xml_path = "data/Kompetenzabgleich.xml"
        if os.path.exists(kompetenzabgleich_xml_path):
            try:
                kompetenzabgleich_data = parse_kompetenzabgleich_xml(kompetenzabgleich_xml_path)
                if kompetenzabgleich_data and kompetenzabgleich_data.get('success'):
                    st.success(f"✅ Kompetenzabgleich XML-Daten erfolgreich geladen: {len(kompetenzabgleich_data.get('ist_rollen', []))} IST-Rollen, {len(kompetenzabgleich_data.get('soll_skills', []))} SOLL-Skills")
                else:
                    st.warning("⚠️ Kompetenzabgleich XML-Daten konnten nicht geladen werden")
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Kompetenzabgleich XML-Daten: {str(e)}")
                kompetenzabgleich_data = None
        else:
            st.warning(f"⚠️ Kompetenzabgleich XML-Datei nicht gefunden: {kompetenzabgleich_xml_path}")
            st.write("**Erwarteter Pfad:** data/Kompetenzabgleich.xml")
        
        return (employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, 
                eures_skills_df, udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df, archi_data, kompetenzabgleich_data)
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {str(e)}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), None, None)

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
    """Findet passende Udemy-Kurse für fehlende Skills (vereinfachte Version)"""
    
    try:
        if not missing_skills or udemy_courses_df.empty:
            return []
        
        # Prüfe ob benötigte Spalten vorhanden sind
        required_columns = ['Title', 'Headline', 'Description', 'URL', 'Price', 'Language']
        missing_columns = [col for col in required_columns if col not in udemy_courses_df.columns]
        if missing_columns:
            st.warning(f"Fehlende Spalten in Udemy-Daten: {missing_columns}")
            return []
        
        recommendations = []
        
        for skill in missing_skills:
            # Vereinfachte Text-Suche ohne TF-IDF
            skill_text = str(skill).lower()
            
            # Suche nach Kursen die den Skill-Text enthalten
            matching_courses = []
            for _, course in udemy_courses_df.iterrows():
                title = str(course.get('Title', '')).lower()
                headline = str(course.get('Headline', '')).lower()
                description = str(course.get('Description', '')).lower()
                
                # Einfache Text-Übereinstimmung
                if (skill_text in title or skill_text in headline or skill_text in description):
                    matching_courses.append({
                        'skill': str(skill),
                        'course_title': course.get('Title', 'N/A'),
                        'course_headline': course.get('Headline', 'N/A'),
                        'course_description': str(course.get('Description', ''))[:200] + '...' if len(str(course.get('Description', ''))) > 200 else course.get('Description', ''),
                        'course_url': course.get('URL', ''),
                        'course_price': course.get('Price', 'N/A'),
                        'course_language': course.get('Language', 'N/A'),
                        'similarity_score': 0.8  # Fester Score für Text-Matches
                    })
            
            # Top-K Kurse pro Skill
            top_courses = matching_courses[:top_k]
            if top_courses:
                recommendations.extend(top_courses)
            else:
                # Fallback: Zufällige Kurse wenn keine Matches gefunden wurden
                random_courses = udemy_courses_df.sample(min(top_k, len(udemy_courses_df))).to_dict('records')
                for course in random_courses:
                    recommendations.append({
                        'skill': str(skill),
                        'course_title': course.get('Title', 'N/A'),
                        'course_headline': course.get('Headline', 'N/A'),
                        'course_description': str(course.get('Description', ''))[:200] + '...' if len(str(course.get('Description', ''))) > 200 else course.get('Description', ''),
                        'course_url': course.get('URL', ''),
                        'course_price': course.get('Price', 'N/A'),
                        'course_language': course.get('Language', 'N/A'),
                        'similarity_score': 0.5  # Niedriger Score für Fallback-Kurse
                    })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Fehler in find_udemy_courses_for_skills: {str(e)}")
        return []

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

@st.cache_data
def parse_archi_xml(xml_file_path):
    """Parst XML-Dateien aus Archi und extrahiert Capabilities und deren zugehörige Skills"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Namespace-Mapping für Archimate
        namespaces = {
            'archimate': 'http://www.opengroup.org/xsd/archimate/3.0/',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
        
        capabilities = []
        resources = []
        
        # Extrahiere alle Capabilities mit korrektem Namespace
        capability_elements = root.findall('.//archimate:element[@xsi:type="Capability"]', namespaces)
        
        # Verarbeite gefundene Capabilities
        for capability in capability_elements:
            cap_id = capability.get('identifier')
            cap_name = capability.find('archimate:name', namespaces)
            if cap_name is not None and cap_name.text:
                capabilities.append({
                    'id': cap_id,
                    'name': cap_name.text,
                    'type': 'Capability'
                })
        
        # Extrahiere alle Resources (Skills) mit korrektem Namespace
        resource_elements = root.findall('.//archimate:element[@xsi:type="Resource"]', namespaces)
        
        # Verarbeite gefundene Resources
        for resource in resource_elements:
            res_id = resource.get('identifier')
            res_name = resource.find('archimate:name', namespaces)
            if res_name is not None and res_name.text:
                resources.append({
                    'id': res_id,
                    'name': res_name.text,
                    'type': 'Resource'
                })
        
        # Extrahiere Beziehungen zwischen Capabilities und Resources
        relationships = []
        relationship_elements = root.findall('.//archimate:relationship', namespaces)
        
        for relation in relationship_elements:
            source = relation.get('source')
            target = relation.get('target')
            rel_type = relation.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            
            if source and target and rel_type:
                relationships.append({
                    'source': source,
                    'target': target,
                    'type': rel_type
                })
        
        # Fallback: Falls keine Elemente gefunden wurden, versuche alternative Strategien
        if not capabilities and not resources:
            st.write("⚠️ Keine Elemente mit Namespace gefunden, versuche alternative Strategien...")
            
            # Alternative 1: Suche nach allen Elementen mit xsi:type
            for element in root.findall('.//*'):
                xsi_type = element.get('{http://www.w3.org/2001/XMLSchema-instance}type')
                if xsi_type == 'Capability':
                    cap_id = element.get('identifier')
                    cap_name = element.find('name')
                    if cap_name is not None and cap_name.text:
                        capabilities.append({
                            'id': cap_id,
                            'name': cap_name.text,
                            'type': 'Capability'
                        })
                elif xsi_type == 'Resource':
                    res_id = element.get('identifier')
                    res_name = element.find('name')
                    if res_name is not None and res_name.text:
                        resources.append({
                            'id': res_id,
                            'name': res_name.text,
                            'type': 'Resource'
                        })
        
        return {
            'capabilities': capabilities,
            'resources': resources,
            'relationships': relationships
        }
        
    except Exception as e:
        st.error(f"Fehler beim Parsen der XML-Datei: {str(e)}")
        st.write("**Debug-Informationen:**")
        st.write(f"• Fehlertyp: {type(e).__name__}")
        st.write(f"• Fehlermeldung: {str(e)}")
        st.write("**Versuche alternative Parsing-Strategie...**")
        
        # Alternative: Einfaches Parsen ohne Namespace-Behandlung
        try:
            # Lade XML als Text und suche nach Schlüsselwörtern
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Einfache Text-basierte Extraktion
            capabilities = []
            resources = []
            
            # Suche nach Capability-Elementen
            import re
            
            # Verschiedene Patterns für Capabilities
            cap_patterns = [
                r'<element[^>]*xsi:type="Capability"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<element[^>]*type="Capability"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<capability[^>]*>.*?<name[^>]*>(.*?)</name>'
            ]
            
            for pattern in cap_patterns:
                cap_matches = re.findall(pattern, xml_content, re.DOTALL)
                for match in cap_matches:
                    clean_name = match.strip()
                    if clean_name and clean_name not in [cap['name'] for cap in capabilities]:
                        capabilities.append({
                            'id': f"cap_{len(capabilities)}",
                            'name': clean_name,
                            'type': 'Capability'
                        })
            
            # Verschiedene Patterns für Resources
            res_patterns = [
                r'<element[^>]*xsi:type="Resource"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<element[^>]*type="Resource"[^>]*>.*?<name[^>]*>(.*?)</name>',
                r'<resource[^>]*>.*?<name[^>]*>(.*?)</name>'
            ]
            
            for pattern in res_patterns:
                res_matches = re.findall(pattern, xml_content, re.DOTALL)
                for match in res_matches:
                    clean_name = match.strip()
                    if clean_name and clean_name not in [res['name'] for res in resources]:
                        resources.append({
                            'id': f"res_{len(resources)}",
                            'name': clean_name,
                            'type': 'Resource'
                        })
            
            st.success(f"✅ Alternative Parsing-Strategie erfolgreich: {len(capabilities)} Capabilities, {len(resources)} Resources gefunden")
            
            return {
                'capabilities': capabilities,
                'resources': resources,
                'relationships': []
            }
            
        except Exception as e2:
            st.error(f"❌ Auch alternative Parsing-Strategie fehlgeschlagen: {str(e2)}")
            return None

@st.cache_data
def extract_future_skills_from_capabilities(archi_data):
    """Extrahiert zukünftig benötigte Skills aus den Capabilities der XML"""
    if not archi_data:
        return []
    
    future_skills = []
    
    # Sammle alle Resources (Skills) aus den Capabilities
    for resource in archi_data['resources']:
        skill_name = resource['name']
        
        # Prüfe ob der Skill mit einer Capability verbunden ist
        is_capability_skill = False
        for relation in archi_data['relationships']:
            if (relation['source'] == resource['id'] or relation['target'] == resource['id']) and \
               any(rel['type'] in ['Composition', 'Aggregation', 'Realization'] for rel in [relation]):
                is_capability_skill = True
                break
        
        if is_capability_skill:
            future_skills.append({
                'skill_name': skill_name,
                'source': 'Capability',
                'type': 'Future Skill'
            })
    
    return future_skills

@st.cache_data
def parse_kompetenzabgleich_xml(xml_file_path):
    """Parst die Kompetenzabgleich.xml und extrahiert IST-Rollen und SOLL-Skills"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        ist_rollen = []  # BusinessActor (aktuelle Rollen)
        soll_skills = []  # Capability (zukünftige Skills)
        
        # Strategie 1: Suche nach allen Elementen und prüfe den xsi:type
        for element in root.findall('.//element'):
            element_type = element.get('xsi:type', '')
            name_elem = element.find('name')
            
            if name_elem is not None and name_elem.text:
                name = name_elem.text
                identifier = element.get('identifier')
                
                if 'BusinessActor' in element_type:
                    ist_rollen.append({
                        'identifier': identifier,
                        'name': name,
                        'type': 'BusinessActor'
                    })
                elif 'Capability' in element_type:
                    soll_skills.append({
                        'identifier': identifier,
                        'name': name,
                        'type': 'Capability'
                    })
        
        # Strategie 2: Fallback - Suche nach Elementen mit bestimmten Tags
        if not ist_rollen and not soll_skills:
            # Suche nach BusinessActor
            for element in root.findall('.//element'):
                if 'BusinessActor' in str(element.attrib):
                    name_elem = element.find('name')
                    if name_elem is not None and name_elem.text:
                        ist_rollen.append({
                            'identifier': element.get('identifier', ''),
                            'name': name_elem.text,
                            'type': 'BusinessActor'
                        })
            
            # Suche nach Capability
            for element in root.findall('.//element'):
                if 'Capability' in str(element.attrib):
                    name_elem = element.find('name')
                    if name_elem is not None and name_elem.text:
                        soll_skills.append({
                            'identifier': element.get('identifier', ''),
                            'name': name_elem.text,
                            'type': 'Capability'
                        })
        
        # Strategie 3: Regex-basierte Suche als letzter Fallback
        if not ist_rollen and not soll_skills:
            import re
            
            # Lese den gesamten XML-Inhalt
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Suche nach BusinessActor-Elementen
            business_actor_pattern = r'<element[^>]*xsi:type="BusinessActor"[^>]*>.*?<name[^>]*>([^<]+)</name>'
            business_actor_matches = re.findall(business_actor_pattern, xml_content, re.DOTALL)
            
            for i, name in enumerate(business_actor_matches):
                ist_rollen.append({
                    'identifier': f'auto_gen_{i}',
                    'name': name.strip(),
                    'type': 'BusinessActor'
                })
            
            # Suche nach Capability-Elementen
            capability_pattern = r'<element[^>]*xsi:type="Capability"[^>]*>.*?<name[^>]*>([^<]+)</name>'
            capability_matches = re.findall(capability_pattern, xml_content, re.DOTALL)
            
            for i, name in enumerate(capability_matches):
                soll_skills.append({
                    'identifier': f'auto_gen_{i}',
                    'name': name.strip(),
                    'type': 'Capability'
                })
        
        # Debug-Informationen
        st.info(f"Debug: Gefundene Elemente - BusinessActor: {len(ist_rollen)}, Capability: {len(soll_skills)}")
        
        return {
            'ist_rollen': ist_rollen,
            'soll_skills': soll_skills,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Fehler beim Parsen der Kompetenzabgleich.xml: {str(e)}")
        st.info("Versuche alternative Parsing-Strategien...")
        
        try:
            # Alternative Strategie: Einfaches Text-basiertes Parsing
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            ist_rollen = []
            soll_skills = []
            
            # Suche nach BusinessActor
            if 'BusinessActor' in xml_content:
                lines = xml_content.split('\n')
                for i, line in enumerate(lines):
                    if 'BusinessActor' in line and i + 1 < len(lines):
                        # Suche nach dem nächsten name-Element
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '<name' in lines[j] and '</name>' in lines[j]:
                                name = lines[j].split('<name')[1].split('</name>')[0].split('>', 1)[1]
                                if name.strip():
                                    ist_rollen.append({
                                        'identifier': f'fallback_{len(ist_rollen)}',
                                        'name': name.strip(),
                                        'type': 'BusinessActor'
                                    })
                                break
            
            # Suche nach Capability
            if 'Capability' in xml_content:
                lines = xml_content.split('\n')
                for i, line in enumerate(lines):
                    if 'Capability' in line and i + 1 < len(lines):
                        # Suche nach dem nächsten name-Element
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '<name' in lines[j] and '</name>' in lines[j]:
                                name = lines[j].split('<name')[1].split('</name>')[0].split('>', 1)[1]
                                if name.strip():
                                    soll_skills.append({
                                        'identifier': f'fallback_{len(soll_skills)}',
                                        'name': name.strip(),
                                        'type': 'Capability'
                                    })
                                break
            
            st.success(f"Fallback-Parsing erfolgreich: {len(ist_rollen)} IST-Rollen, {len(soll_skills)} SOLL-Skills")
            
            return {
                'ist_rollen': ist_rollen,
                'soll_skills': soll_skills,
                'success': True
            }
            
        except Exception as fallback_error:
            st.error(f"Auch Fallback-Parsing fehlgeschlagen: {str(fallback_error)}")
            return {
                'ist_rollen': [],
                'soll_skills': [],
                'success': False,
                'error': f"Original: {str(e)}, Fallback: {str(fallback_error)}"
            }

@st.cache_data
def find_kldb_code_for_job_title(job_title, occupations_df, kldb_esco_df):
    """Findet den passenden KldB-Code für eine Jobbezeichnung"""
    if not job_title or occupations_df.empty:
        return None, None
    
    # Normalisiere den Jobtitel
    normalized_job = job_title.lower().strip()
    
    # Suche nach exakten Matches in den Bezeichnungen
    for _, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', '')).lower()
        if normalized_job in preferred_label or preferred_label in normalized_job:
            # Finde den zugehörigen KldB-Code
            concept_uri = str(row.get('conceptUri', ''))
            preferred_label = str(row.get('preferredLabel', ''))
            if concept_uri and preferred_label:
                # Versuche zuerst direktes Label-Matching mit KldB-ESCO Mapping
                for _, kldb_row in kldb_esco_df.iterrows():
                    esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                    if preferred_label.lower() == esco_label or preferred_label.lower() in esco_label:
                        kldb_code = kldb_row.get('KldB_Code', '')
                        kldb_label = kldb_row.get('KldB_Label', '')
                        if kldb_code:
                            return kldb_code, kldb_label
    
    # Fallback: Suche nach ähnlichen Bezeichnungen
    best_match = None
    best_score = 0
    
    for _, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', '')).lower()
        
        # Einfache Ähnlichkeitsberechnung
        common_words = set(normalized_job.split()) & set(preferred_label.split())
        if common_words:
            score = len(common_words) / max(len(normalized_job.split()), len(preferred_label.split()))
            if score > best_score and score > 0.3:  # Mindestähnlichkeit
                best_score = score
                best_match = row
    
    if best_match is not None:
        preferred_label = str(best_match.get('preferredLabel', ''))
        if preferred_label:
            # Versuche Label-Matching mit KldB-ESCO Mapping
            for _, kldb_row in kldb_esco_df.iterrows():
                esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                if preferred_label.lower() in esco_label or esco_label in preferred_label.lower():
                    kldb_code = kldb_row.get('KldB_Code', '')
                    kldb_label = kldb_row.get('KldB_Label', '')
                    if kldb_code:
                        return kldb_code, kldb_label
    
    return None, None

@st.cache_data
def find_best_job_matches_for_capabilities(capabilities, occupations_df, kldb_esco_df, top_k=5):
    """Findet die besten Jobtitel-Matches für Capabilities (SOLL-Skills)"""
    if not capabilities or occupations_df.empty:
        return []
    
    # TF-IDF Vektorisierung für Jobbeschreibungen
    job_descriptions = []
    job_indices = []
    
    for idx, row in occupations_df.iterrows():
        preferred_label = str(row.get('preferredLabel', ''))
        if preferred_label and not pd.isna(preferred_label):
            job_descriptions.append(preferred_label)
            job_indices.append(idx)
    
    if not job_descriptions:
        return []
    
    # TF-IDF Vektorisierung
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    job_vectors = vectorizer.fit_transform(job_descriptions)
    
    all_matches = []
    
    for capability in capabilities:
        capability_name = capability.get('name', '')
        if not capability_name:
            continue
        
        # Vektorisierung der Capability
        capability_vector = vectorizer.transform([capability_name])
        
        # Ähnlichkeitsberechnung
        similarities = cosine_similarity(capability_vector, job_vectors).flatten()
        
        # Top-K Matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        capability_matches = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0.01:  # Mindestähnlichkeit
                job_idx = job_indices[idx]
                job_row = occupations_df.iloc[job_idx]
                
                # Finde KldB-Code
                preferred_label = str(job_row.get('preferredLabel', ''))
                kldb_code = ''
                kldb_label = ''
                
                if preferred_label:
                    # Versuche Label-Matching mit KldB-ESCO Mapping
                    for _, kldb_row in kldb_esco_df.iterrows():
                        esco_label = str(kldb_row.get('ESCO_Label', '')).lower()
                        if preferred_label.lower() in esco_label or esco_label in preferred_label.lower():
                            kldb_code = kldb_row.get('KldB_Code', '')
                            kldb_label = kldb_row.get('KldB_Label', '')
                            break
                
                capability_matches.append({
                    'capability': capability_name,
                    'job_title': job_row.get('preferredLabel', ''),
                    'kldb_code': kldb_code,
                    'kldb_label': kldb_label,
                    'esco_uri': job_row.get('conceptUri', ''),
                    'similarity_score': similarities[idx],
                    'rank': rank + 1
                })
        
        if capability_matches:
            all_matches.extend(capability_matches)
    
    # Sortiere nach Ähnlichkeits-Score
    all_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return all_matches

def main():
    st.title("Kompetenzabgleich & Weiterbildungsempfehlungen")
    st.markdown("---")
    
    # Lade Daten
    with st.spinner("Lade Daten..."):
        employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, udemy_courses_df, occupations_df, occupation_skills_mapping, skills_en_df, archi_data, kompetenzabgleich_data = load_data()
    
    if employees_df.empty and kldb_esco_df.empty:
        st.error("Fehler beim Laden der Daten. Bitte überprüfe die CSV-Dateien.")
        return
    
    # Erstelle das direkte Skills-Mapping
    st.session_state.occupation_skills_mapping = occupation_skills_mapping
    
    # Erstelle das Skill-Mapping mit englischen Entsprechungen
    skill_mapping_with_english = create_skill_mapping_with_english(skills_df, skills_en_df)
    st.session_state.skill_mapping_with_english = skill_mapping_with_english
    
    # Speichere Archi-Daten im Session State
    st.session_state.archi_data = archi_data
    
    # Speichere Kompetenzabgleich-Daten im Session State
    st.session_state.kompetenzabgleich_data = kompetenzabgleich_data
    
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
        ["Mitarbeiter-Kompetenzprofile", "Berufsabgleich", "Strategische Weiterbildung 🆕", "XML-basierte Kompetenzabgleich 🆕", "Kursempfehlungen", "Gesamtübersicht", "Mitarbeiter-Verwaltung"]
    )
    
    # Zeige entsprechende Seite
    if page == "Mitarbeiter-Kompetenzprofile":
        show_employee_profiles(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df)
    elif page == "Berufsabgleich":
        show_occupation_matching(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df)
    elif page == "Strategische Weiterbildung 🆕":
        show_strategic_development(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, archi_data, udemy_courses_df)
    elif page == "XML-basierte Kompetenzabgleich 🆕":
        show_xml_based_competency_analysis(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df, udemy_courses_df)
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
        if st.button("Zielrolle zurücksetzen", key="reset_target_role_profile", type="secondary"):
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
                    
                    # Individuelle Checkbox für jeden Skill mit eindeutigem Key
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_essential_{i}_{skill['skill_uri']}",
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
                    
                    # Individuelle Checkbox für jeden Skill mit eindeutigem Key
                    is_checked = st.checkbox(
                        f"Entfernen: {skill['skill_label']}", 
                        key=f"remove_optional_{i}_{skill['skill_uri']}",
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
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
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
        
        st.session_state.employees_data = employees_df.copy() if not employees_df.empty else pd.DataFrame(columns=['Employee_ID', 'Name', 'KldB_5_digit', 'Manual_Skills', 'ESCO_Role', 'Target_KldB_Code', 'Target_KldB_Label', 'Target_ESCO_Code', 'Target_ESCO_Label', 'Manual_Essential_Skills', 'Manual_Optional_Skills', 'Removed_Skills'])
    
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
    
    if st.button("Exportiere alle Ergebnisse als CSV", key="export_all_results_csv"):
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
                    if st.button("Endgültig löschen", key="delete_employee_confirm", type="primary"):
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
                    if st.button("Abbrechen", key="cancel_delete_employee"):
                        st.rerun()
    
    # Aktualisiere die globale employees_df für andere Sektionen
    if 'employees_data' in st.session_state:
        globals()['employees_df'] = st.session_state.employees_data

def show_strategic_development(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, archi_data, udemy_courses_df):
    st.header("Strategische Weiterbildung 🆕")
    st.info("**Neue Funktionalität:** Diese Sektion nutzt XML-Daten aus Archi, um strategische Weiterbildungsempfehlungen basierend auf Geschäftsmodellen zu generieren.")
    
    # Debug-Informationen anzeigen
    st.write("**Debug-Informationen:**")
    st.write(f"• Archi-Daten verfügbar: {archi_data is not None}")
    if archi_data:
        st.write(f"• Capabilities gefunden: {len(archi_data.get('capabilities', []))}")
        st.write(f"• Resources gefunden: {len(archi_data.get('resources', []))}")
        st.write(f"• Beziehungen gefunden: {len(archi_data.get('relationships', []))}")
    
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
    
    st.subheader(f"Strategische Weiterbildung für {employee_data.get('Name', employee_id)}")
    
    # Prüfe ob Archi-Daten verfügbar sind
    if not archi_data:
        st.error("Keine Archi XML-Daten verfügbar. Bitte stellen Sie sicher, dass die DigiVan.xml Datei im data-Ordner liegt.")
        st.write("**Mögliche Ursachen:**")
        st.write("• Die DigiVan.xml Datei fehlt im data-Ordner")
        st.write("• Die XML-Datei konnte nicht geparst werden")
        st.write("• Ein Fehler ist beim Laden der Daten aufgetreten")
        
        # Versuche die XML-Datei manuell zu laden
        xml_path = 'data/DigiVan.xml'
        if os.path.exists(xml_path):
            st.write(f"✅ XML-Datei gefunden: {xml_path}")
            st.write(f"📁 Dateigröße: {os.path.getsize(xml_path)} Bytes")
            
            # Versuche manuelles Parsen
            if st.button("XML-Datei manuell neu laden", key="reload_xml_manual"):
                try:
                    manual_archi_data = parse_archi_xml(xml_path)
                    if manual_archi_data:
                        st.session_state.archi_data = manual_archi_data
                        st.success("XML-Datei erfolgreich geladen!")
                        st.rerun()
                    else:
                        st.error("Manuelles Laden fehlgeschlagen")
                except Exception as e:
                    st.error(f"Fehler beim manuellen Laden: {str(e)}")
        else:
            st.error(f"❌ XML-Datei nicht gefunden: {xml_path}")
        
        return
    
    # Aktuelle Rolle des Mitarbeiters
    current_kldb = employee_data.get('KldB_5_digit', '')
    current_manual_skills = employee_data.get('Manual_Skills', '')
    current_esco_role = employee_data.get('ESCO_Role', '')
    
    # Hole zusätzliche Skill-Daten aus Session State
    current_manual_essential_skills = employee_data.get('Manual_Essential_Skills', '')
    current_manual_optional_skills = employee_data.get('Manual_Optional_Skills', '')
    current_removed_skills = employee_data.get('Removed_Skills', '')
    
    if not current_kldb:
        st.info("**Keine aktuelle Rolle zugewiesen.** Bitte weisen Sie zuerst eine Rolle in 'Mitarbeiter-Kompetenzprofile' zu.")
        return
    
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
        current_esco_role,
        current_manual_essential_skills,
        current_manual_optional_skills,
        current_removed_skills
    )
    
    if not current_profile:
        st.error("Konnte kein aktuelles Mitarbeiterprofil erstellen.")
        return
    
    # Extrahiere zukünftig benötigte Skills aus den Capabilities
    future_skills = extract_future_skills_from_capabilities(archi_data)
    
    if not future_skills:
        st.warning("Keine zukünftig benötigten Skills aus den Capabilities gefunden.")
        return
    
    # Zeige aktuelle und zukünftige Skills
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aktuelle Skills:**")
        current_skills = current_profile['skills']
        st.write(f"• Anzahl: {len(current_skills)}")
        
        # Gruppiere aktuelle Skills
        current_essential = [s for s in current_skills if s.get('is_essential', False)]
        current_optional = [s for s in current_skills if not s.get('is_essential', False)]
        
        if current_essential:
            st.write("**Essential Skills:**")
            for skill in current_essential[:5]:  # Zeige nur die ersten 5
                st.write(f"  - {skill['skill_label']}")
            if len(current_essential) > 5:
                st.write(f"  ... und {len(current_essential) - 5} weitere")
        
        if current_optional:
            st.write("**Optional Skills:**")
            for skill in current_optional[:5]:  # Zeige nur die ersten 5
                st.write(f"  - {skill['skill_label']}")
            if len(current_optional) > 5:
                st.write(f"  ... und {len(current_optional) - 5} weitere")
    
    with col2:
        st.write("**Zukünftig benötigte Skills (aus Capabilities):**")
        st.write(f"• Anzahl: {len(future_skills)}")
    
    st.markdown("---")
    
    # Strategischer Kompetenzabgleich
    st.subheader("Strategischer Kompetenzabgleich")
    
    # Vergleiche aktuelle Skills mit zukünftig benötigten Skills
    current_skill_labels = [skill['skill_label'].lower() for skill in current_skills]
    future_skill_labels = [skill['skill_name'].lower() for skill in future_skills]
    
    # Finde übereinstimmende Skills
    matching_skills = []
    missing_skills = []
    
    for future_skill in future_skills:
        future_label = future_skill['skill_name'].lower()
        
        # Suche nach exakten Matches
        exact_match = None
        for current_skill in current_skills:
            if current_skill['skill_label'].lower() == future_label:
                exact_match = current_skill
                break
        
        if exact_match:
            matching_skills.append({
                'future_skill': future_skill,
                'current_skill': exact_match,
                'match_type': 'exakt'
            })
        else:
            # Suche nach ähnlichen Skills (semantisches Matching)
            similar_skill = None
            for current_skill in current_skills:
                if any(word in current_skill['skill_label'].lower() for word in future_label.split()):
                    similar_skill = current_skill
                    break
            
            if similar_skill:
                matching_skills.append({
                    'future_skill': future_skill,
                    'current_skill': similar_skill,
                    'match_type': 'ähnlich'
                })
            else:
                missing_skills.append(future_skill)
    
    # Zeige Ergebnisse
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Übereinstimmende Skills:**")
        st.write(f"• Anzahl: {len(matching_skills)}")
        
        if matching_skills:
            for match in matching_skills[:5]:
                match_type_icon = "✅" if match['match_type'] == 'exakt' else "🔄"
                st.write(f"{match_type_icon} {match['future_skill']['skill_name']}")
                if match['match_type'] == 'ähnlich':
                    st.write(f"  → Ähnlich zu: {match['current_skill']['skill_label']}")
            if len(matching_skills) > 5:
                st.write(f"  ... und {len(matching_skills) - 5} weitere")
    
    with col2:
        st.write("**Fehlende Skills für digitale Transformation:**")
        st.write(f"• Anzahl: {len(missing_skills)}")
        
        if missing_skills:
            for skill in missing_skills[:5]:
                st.write(f"❌ {skill['skill_name']}")
            if len(missing_skills) > 5:
                st.write(f"  ... und {len(missing_skills) - 5} weitere")
    
    st.markdown("---")
    
    # Strategische Empfehlungen
    st.subheader("Strategische Empfehlungen")
    
    if missing_skills:
        st.write("**Prioritäre Weiterbildungsbereiche:**")
        
        # Gruppiere fehlende Skills nach Kategorien
        skill_categories = {}
        for skill in missing_skills:
            skill_name = skill['skill_name'].lower()
            
            # Einfache Kategorisierung basierend auf Schlüsselwörtern
            if any(word in skill_name for word in ['data', 'analytics', 'mining', 'visualization']):
                category = "Datenanalyse & Business Intelligence"
            elif any(word in skill_name for word in ['python', 'sql', 'programming', 'coding']):
                category = "Programmierung & Technische Skills"
            elif any(word in skill_name for word in ['customer', 'service', 'communication', 'relationship']):
                category = "Kundenorientierung & Kommunikation"
            elif any(word in skill_name for word in ['maintenance', 'equipment', 'troubleshoot', 'technical']):
                category = "Technische Wartung & Problemlösung"
            else:
                category = "Sonstige Skills"
            
            if category not in skill_categories:
                skill_categories[category] = []
            skill_categories[category].append(skill)
        
        # Zeige kategorisierte Empfehlungen
        for category, skills in skill_categories.items():
            with st.expander(f"{category} ({len(skills)} Skills)"):
                for skill in skills:
                    st.write(f"• {skill['skill_name']}")
                
                # Generiere semantische Kursempfehlungen für diese Kategorie
                st.write("**🎯 Semantische Kursempfehlungen von Udemy:**")
                
                # Erstelle Skill-Objekte für die Kursempfehlung
                skill_objects = []
                for skill in skills:
                    skill_objects.append({
                        'skill_label': skill['skill_name'],
                        'skill_uri': f"strategic_{skill['skill_name'].lower().replace(' ', '_')}",
                        'is_essential': True
                    })
                
                # Finde Kursempfehlungen für diese Skills
                with st.spinner(f"Suche Kurse für {category}..."):
                    try:
                        # Verwende die übergebenen Udemy-Daten
                        if udemy_courses_df is not None and not udemy_courses_df.empty:
                            
                            # Finde passende Kurse
                            recommendations = find_udemy_courses_for_skills(
                                skill_objects,
                                udemy_courses_df,
                                top_k=3  # Zeige Top 3 Kurse pro Kategorie
                            )
                            
                            if recommendations:
                                st.write(f"**Top {len(recommendations)} Kurse gefunden:**")
                                
                                for i, course in enumerate(recommendations[:3], 1):
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.write(f"**{i}. {course['course_title']}**")
                                            if course.get('course_headline'):
                                                st.write(f"*{course['course_headline']}*")
                                            if course.get('course_description'):
                                                # Kürze die Beschreibung
                                                desc = course['course_description'][:200] + "..." if len(course['course_description']) > 200 else course['course_description']
                                                st.write(f"*{desc}*")
                                            
                                            # Zeige relevante Skills
                                            if course.get('skill'):
                                                st.write("**Relevanter Skill:**")
                                                st.write(f"• {course['skill']}")
                                            
                                            # Zeige Ähnlichkeits-Score
                                            if course.get('similarity_score'):
                                                st.write(f"**Relevanz:** {course['similarity_score']:.2f}")
                                        
                                        with col2:
                                            if course.get('course_price'):
                                                st.write(f"**Preis:** {course['course_price']}")
                                            if course.get('course_language'):
                                                st.write(f"**Sprache:** {course['course_language']}")
                                            if course.get('course_url'):
                                                st.write(f"[Kurs öffnen]({course['course_url']})")
                                        
                                        st.markdown("---")
                            else:
                                st.info("Keine spezifischen Kurse für diese Kategorie gefunden.")
                                st.write("**Allgemeine Empfehlungen:**")
                                
                                # Fallback-Empfehlungen basierend auf der Kategorie
                                if category == "Datenanalyse & Business Intelligence":
                                    st.write("• Kurse in Data Analytics und Business Intelligence")
                                    st.write("• Schulungen in Datenvisualisierung")
                                    st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                                elif category == "Programmierung & Technische Skills":
                                    st.write("• Python-Programmierkurse")
                                    st.write("• SQL-Datenbankkurse")
                                    st.write("• Einführung in maschinelles Lernen")
                                elif category == "Kundenorientierung & Kommunikation":
                                    st.write("• Kundenservice-Schulungen")
                                    st.write("• Kommunikationstraining")
                                    st.write("• Beziehungsmanagement")
                                elif category == "Technische Wartung & Problemlösung":
                                    st.write("• Wartungsverfahren und -standards")
                                    st.write("• Problemlösungstechniken")
                                    st.write("• Technische Dokumentation")
                                else:
                                    st.write("• Allgemeine Weiterbildungsmaßnahmen")
                                    st.write("• Spezifische Schulungen je nach Skill")
                                    st.write("• On-the-Job Training")
                        else:
                            st.warning("Udemy-Kursdaten nicht verfügbar. Verwende allgemeine Empfehlungen.")
                            # Fallback-Empfehlungen
                            if category == "Datenanalyse & Business Intelligence":
                                st.write("• Kurse in Data Analytics und Business Intelligence")
                                st.write("• Schulungen in Datenvisualisierung")
                                st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                            elif category == "Programmierung & Technische Skills":
                                st.write("• Python-Programmierkurse")
                                st.write("• SQL-Datenbankkurse")
                                st.write("• Einführung in maschinelles Lernen")
                            elif category == "Kundenorientierung & Kommunikation":
                                st.write("• Kundenservice-Schulungen")
                                st.write("• Kommunikationstraining")
                                st.write("• Beziehungsmanagement")
                            elif category == "Technische Wartung & Problemlösung":
                                st.write("• Wartungsverfahren und -standards")
                                st.write("• Problemlösungstechniken")
                                st.write("• Technische Dokumentation")
                            else:
                                st.write("• Allgemeine Weiterbildungsmaßnahmen")
                                st.write("• Spezifische Schulungen je nach Skill")
                                st.write("• On-the-Job Training")
                                
                    except Exception as e:
                        st.error(f"Fehler bei der Kursempfehlung: {str(e)}")
                        st.write("**Verwende allgemeine Empfehlungen:**")
                        if category == "Datenanalyse & Business Intelligence":
                            st.write("• Kurse in Data Analytics und Business Intelligence")
                            st.write("• Schulungen in Datenvisualisierung")
                            st.write("• Workshops zu datengetriebener Entscheidungsfindung")
                        elif category == "Programmierung & Technische Skills":
                            st.write("• Python-Programmierkurse")
                            st.write("• SQL-Datenbankkurse")
                            st.write("• Einführung in maschinelles Lernen")
                        elif category == "Kundenorientierung & Kommunikation":
                            st.write("• Kundenservice-Schulungen")
                            st.write("• Kommunikationstraining")
                            st.write("• Beziehungsmanagement")
                        elif category == "Technische Wartung & Problemlösung":
                            st.write("• Wartungsverfahren und -standards")
                            st.write("• Problemlösungstechniken")
                            st.write("• Technische Dokumentation")
                        else:
                            st.write("• Allgemeine Weiterbildungsmaßnahmen")
                            st.write("• Spezifische Schulungen je nach Skill")
                            st.write("• On-the-Job Training")
    else:
        st.success("**Alle zukünftig benötigten Skills sind bereits vorhanden!**")
        st.write("Der Mitarbeiter ist bereits gut auf die digitale Transformation vorbereitet.")
    
    # Export der strategischen Analyse
    st.markdown("---")
    st.subheader("Export der strategischen Analyse")
    
    if st.button("Strategische Analyse als CSV exportieren", key="export_strategic_analysis_csv"):
        # Erstelle DataFrame für Export
        analysis_data = []
        
        for skill in future_skills:
            skill_name = skill['skill_name']
            status = "Vorhanden" if any(s['skill_name'].lower() == skill_name.lower() for s in matching_skills) else "Fehlt"
            match_type = next((m['match_type'] for m in matching_skills if m['future_skill']['skill_name'].lower() == skill_name.lower()), "Kein Match")
            
            analysis_data.append({
                'Zukünftig_benötigter_Skill': skill_name,
                'Status': status,
                'Match_Typ': match_type,
                'Quelle': 'Capability Map'
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # CSV-Download
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"strategische_analyse_{employee_data.get('Name', employee_id)}_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_xml_based_competency_analysis(employees_df, kldb_esco_df, occupation_skill_relations_df, skills_df, eures_skills_df, occupations_df, skills_en_df, udemy_courses_df):
    """Zeigt die XML-basierte Kompetenzabgleich-Funktionalität"""
    st.header("XML-basierte Kompetenzabgleich 🆕")
    st.info("**Neue Funktionalität:** Diese Sektion nutzt XML-Daten aus Archi, um automatisch IST-Rollen und SOLL-Skills zu extrahieren und Mitarbeiterprofile zu erstellen.")
    
    # Lade beide XML-Dateien
    kompetenzabgleich_path = "data/Kompetenzabgleich.xml"
    digivan_path = "data/DigiVan.xml"
    
    if not os.path.exists(kompetenzabgleich_path):
        st.error(f"Kompetenzabgleich XML-Datei nicht gefunden: {kompetenzabgleich_path}")
        return
    
    if not os.path.exists(digivan_path):
        st.error(f"DigiVan XML-Datei nicht gefunden: {digivan_path}")
        return
    
    # Parse beide XML-Dateien
    with st.spinner("Lade und parse XML-Dateien..."):
        kompetenzabgleich_data = parse_kompetenzabgleich_xml(kompetenzabgleich_path)
        digivan_data = parse_archi_xml(digivan_path)
    
    if not kompetenzabgleich_data['success']:
        st.error("Fehler beim Parsen der Kompetenzabgleich.xml")
        return
    
    # Überprüfe ob DigiVan-Daten erfolgreich geladen wurden
    if not digivan_data or 'capabilities' not in digivan_data:
        st.error("Fehler beim Parsen der DigiVan.xml")
        return
    
    # IST-Rollen aus Kompetenzabgleich.xml
    ist_rollen = kompetenzabgleich_data['ist_rollen']
    
    # SOLL-Skills (Capabilities) aus DigiVan.xml
    soll_skills = digivan_data['capabilities']
    
    st.success(f"✅ Beide XML-Dateien erfolgreich geladen!")
    st.info(f"""
    **📊 XML-Daten:**
    - **IST-Rollen** (Kompetenzabgleich.xml): {len(ist_rollen)} BusinessActor
    - **SOLL-Skills** (DigiVan.xml): {len(soll_skills)} Capabilities
    """)
    
    # Tabs für verschiedene Funktionalitäten
    tab1, tab2 = st.tabs(["📋 IST-Rollen aus XML", "🎯 SOLL-Skills zu Jobs"])
    
    with tab1:
        st.subheader("📋 IST-Rollen aus XML (BusinessActor)")
        
        if ist_rollen:
            st.write(f"**Gefundene IST-Rollen:** {len(ist_rollen)}")
            
            # Tabelle mit IST-Rollen
            ist_data = []
            for rolle in ist_rollen:
                ist_data.append({
                    'Rollenname': rolle['name'],
                    'Typ': rolle['type'],
                    'ID': rolle.get('id', rolle.get('identifier', 'N/A'))
                })
            
            ist_df = pd.DataFrame(ist_data)
            st.dataframe(ist_df, use_container_width=True)
            
            # Neue Funktionalität: IST-Rollen mit KldB-Rollen matchen und ESCO-Rollen zuweisen
            st.markdown("---")
            st.subheader("🎯 IST-Rollen mit KldB-Rollen matchen und ESCO-Rollen zuweisen")
            
            st.write("**Automatisches Matching der gefundenen IST-Rollen mit passenden KldB-Rollen:**")
            
            # Erstelle eine Tabelle mit IST-Rollen und gematchten KldB-Rollen
            ist_kldb_matches = []
            
            for rolle in ist_rollen:
                rolle_name = rolle['name']
                
                # Versuche automatisches Matching mit KldB-Rollen
                kldb_code, kldb_label = find_kldb_code_for_job_title(
                    rolle_name, occupations_df, kldb_esco_df
                )
                
                ist_kldb_matches.append({
                    'IST-Rolle': rolle_name,
                    'Gematcher KldB-Code': kldb_code if kldb_code else 'Nicht gefunden',
                    'Gematcher KldB-Rolle': kldb_label if kldb_label else 'Nicht gefunden',
                    'Matching-Status': '✅ Gefunden' if kldb_code else '❌ Nicht gefunden'
                })
            
            # Zeige Matching-Ergebnisse
            st.write("**Matching-Ergebnisse:**")
            matches_df = pd.DataFrame(ist_kldb_matches)
            st.dataframe(matches_df, use_container_width=True)
            
            # Wähle eine gematchte IST-Rolle für ESCO-Rollen-Zuweisung
            st.markdown("---")
            st.subheader("🔍 ESCO-Rollen für gematchte IST-Rollen auswählen")
            
            # Filtere nur erfolgreich gematchte Rollen
            successful_matches = [match for match in ist_kldb_matches if match['Gematcher KldB-Code'] != 'Nicht gefunden']
            
            if successful_matches:
                st.success(f"✅ {len(successful_matches)} IST-Rollen erfolgreich mit KldB-Rollen gematcht!")
                
                # Dropdown für erfolgreich gematchte Rollen
                match_options = [f"{match['IST-Rolle']} → {match['Gematcher KldB-Rolle']} ({match['Gematcher KldB-Code']})" for match in successful_matches]
                match_options.insert(0, "Bitte wählen Sie eine gematchte IST-Rolle...")
                
                selected_match = st.selectbox(
                    "Wählen Sie eine gematchte IST-Rolle für ESCO-Rollen-Zuweisung:",
                    match_options,
                    key="xml_match_select"
                )
                
                if selected_match and selected_match != "Bitte wählen Sie eine gematchte IST-Rolle...":
                    # Extrahiere KldB-Code aus der Auswahl
                    selected_kldb_code = selected_match.split("(")[1].split(")")[0]
                    
                    # Finde alle zugehörigen ESCO-Rollen für diese KldB-Rolle
                    matching_esco_roles = kldb_esco_df[kldb_esco_df['KldB_Code'] == selected_kldb_code]
                    
                    if not matching_esco_roles.empty:
                        st.write(f"**Verfügbare ESCO-Rollen für KldB-Code '{selected_kldb_code}':**")
                        
                        # Dropdown für ESCO-Rollen Auswahl
                        esco_role_options = [f"{role['ESCO_Label']} ({role['ESCO_Code']})" for _, role in matching_esco_roles.iterrows()]
                        selected_esco_role = st.selectbox(
                            "Wählen Sie eine ESCO-Rolle:",
                            ["Bitte wählen Sie eine ESCO-Rolle..."] + esco_role_options,
                            key=f"esco_role_select_{selected_kldb_code}"
                        )
                        
                        if selected_esco_role and selected_esco_role != "Bitte wählen Sie eine ESCO-Rolle...":
                            # Extrahiere ESCO-Label und Code
                            esco_label = selected_esco_role.split(" (")[0]
                            esco_code = selected_esco_role.split("(")[1].split(")")[0]
                            
                            # Hole Skills für diese ESCO-Rolle
                            role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                            
                            st.write(f"**Ausgewählte ESCO-Rolle:** {esco_label} ({esco_code})")
                            
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
                                if st.button(f"Als aktuelle Rolle übernehmen", key=f"xml_assign_esco_{esco_code}"):
                                    # Aktualisiere den KldB-Code in den Session State Daten
                                    if 'current_employee_id' in st.session_state:
                                        employee_id = st.session_state.current_employee_id
                                        st.session_state.employees_data.loc[
                                            st.session_state.employees_data['Employee_ID'] == employee_id, 
                                            'KldB_5_digit'
                                        ] = selected_kldb_code
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
                                        st.warning("Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
                            else:
                                st.write("Keine Skills für diese Rolle gefunden.")
                        else:
                            st.info("💡 Wählen Sie eine ESCO-Rolle aus, um die Details zu sehen.")
                    else:
                        st.warning(f"Keine ESCO-Rollen für KldB-Code '{selected_kldb_code}' gefunden.")
            else:
                st.warning("❌ Keine IST-Rollen konnten erfolgreich mit KldB-Rollen gematcht werden.")
                st.info("💡 Tipp: Überprüfen Sie die Schreibweise der Rollennamen oder fügen Sie manuell KldB-Codes hinzu.")
            
                        # Manuelle KldB-Zuordnung für nicht gematchte Rollen
            if any(match['Matching-Status'] == '❌ Nicht gefunden' for match in ist_kldb_matches):
                st.markdown("---")
                st.subheader("📝 Manuelle KldB-Zuordnung für nicht gematchte Rollen")
                
                # Zeige nicht gematchte Rollen
                unmatched_roles = [match for match in ist_kldb_matches if match['Matching-Status'] == '❌ Nicht gefunden']
                st.write(f"**Nicht gematchte Rollen:** {len(unmatched_roles)}")
                
                # Dropdown für Rollenauswahl
                selected_unmatched_role = st.selectbox(
                    "Wählen Sie eine nicht gematchte Rolle:",
                    [f"{unmatched['IST-Rolle']}" for unmatched in unmatched_roles],
                    index=0,
                    key="unmatched_role_select"
                )
                
                if selected_unmatched_role:
                    # Finde die ausgewählte Rolle
                    selected_role_data = next((unmatched for unmatched in unmatched_roles if unmatched['IST-Rolle'] == selected_unmatched_role), None)
                    
                    if selected_role_data:
                        st.write(f"**🔍 Ausgewählte IST-Rolle:** {selected_role_data['IST-Rolle']}")
                        
                        # Dropdown für KldB-Code Auswahl mit Codes und Bezeichnungen (ohne Duplikate)
                        available_kldb_options = []
                        seen_combinations = set()  # Verhindert Duplikate
                        
                        for _, row in kldb_esco_df.iterrows():
                            kldb_code = str(row.get('KldB_Code', ''))
                            kldb_label = str(row.get('KldB_Label', ''))
                            if kldb_code and kldb_label and not pd.isna(kldb_code) and not pd.isna(kldb_label):
                                # Kürze lange Labels für bessere Lesbarkeit
                                display_label = kldb_label
                                if len(display_label) > 50:
                                    display_label = display_label[:47] + "..."
                                option = f"{display_label} | {kldb_code}"
                                
                                # Füge nur hinzu, wenn die Kombination noch nicht vorhanden ist
                                if option not in seen_combinations:
                                    available_kldb_options.append(option)
                                    seen_combinations.add(option)
                        
                        # Sortiere nach Bezeichnung für bessere Übersichtlichkeit
                        available_kldb_options = sorted(available_kldb_options)
                        
                        selected_kldb_option = st.selectbox(
                            f"Wählen Sie einen KldB-Code für '{selected_role_data['IST-Rolle']}':",
                            ["Bitte wählen Sie einen KldB-Code..."] + available_kldb_options,
                            key=f"kldb_code_select_{selected_role_data['IST-Rolle']}"
                        )
                        
                        # Extrahiere den KldB-Code aus der ausgewählten Option
                        selected_kldb_code = None
                        if selected_kldb_option and selected_kldb_option != "Bitte wählen Sie einen KldB-Code...":
                            selected_kldb_code = selected_kldb_option.split(" | ")[1]
                        
                        if selected_kldb_code and selected_kldb_code != "Bitte wählen Sie einen KldB-Code...":
                            # Finde zugehörige ESCO-Rollen
                            manual_esco_roles = kldb_esco_df[kldb_esco_df['KldB_Code'] == selected_kldb_code]
                            
                            if not manual_esco_roles.empty:
                                st.success(f"✅ KldB-Code '{selected_kldb_code}' gefunden!")
                                st.write(f"**Verfügbare ESCO-Rollen:**")
                                
                                # Dropdown für ESCO-Rollen Auswahl
                                esco_role_options = [f"{role['ESCO_Label']} ({role['ESCO_Code']})" for _, role in manual_esco_roles.iterrows()]
                                selected_esco_role = st.selectbox(
                                    "Wählen Sie eine ESCO-Rolle:",
                                    ["Bitte wählen Sie eine ESCO-Rolle..."] + esco_role_options,
                                    key=f"esco_role_select_{selected_kldb_code}"
                                )
                                
                                if selected_esco_role and selected_esco_role != "Bitte wählen Sie eine ESCO-Rolle...":
                                    # Extrahiere ESCO-Label und Code
                                    esco_label = selected_esco_role.split(" (")[0]
                                    esco_code = selected_esco_role.split("(")[1].split(")")[0]
                                    
                                    # Hole Skills für diese ESCO-Rolle
                                    role_skills = get_skills_for_occupation_simple(esco_label, st.session_state.occupation_skills_mapping, occupations_df)
                                    
                                    if role_skills:
                                        st.write(f"**Skills für {esco_label}:**")
                                        for skill in role_skills:
                                            essential_mark = " (Essential)" if skill['is_essential'] else " (Optional)"
                                            st.write(f"• {skill['skill_label']}{essential_mark}")
                                        
                                        # Button zum Übernehmen
                                        if st.button(f"Als aktuelle Rolle übernehmen", key=f"manual_assign_esco_{selected_kldb_code}_{esco_code}"):
                                            if 'current_employee_id' in st.session_state:
                                                employee_id = st.session_state.current_employee_id
                                                st.session_state.employees_data.loc[
                                                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                                                    'KldB_5_digit'
                                                ] = selected_kldb_code
                                                st.session_state.employees_data.loc[
                                                    st.session_state.employees_data['Employee_ID'] == employee_id, 
                                                    'ESCO_Role'
                                                ] = esco_label
                                                
                                                # Speichere die Auswahl im Session State für Tab 2
                                                st.session_state.manual_selection_tab1 = f"{selected_role_data['IST-Rolle']} → {esco_label} ({selected_kldb_code})"
                                                st.session_state.selected_kldb_code_tab2 = selected_kldb_code
                                                st.session_state.selected_esco_role_tab2 = esco_label
                                                st.session_state.selected_role_data_name = selected_role_data['IST-Rolle']
                                                
                                                if save_employees_to_csv(st.session_state.employees_data):
                                                    st.success(f"Rolle '{esco_label}' wurde als aktuelle Rolle zugewiesen und gespeichert!")
                                                    st.info("💡 Wechseln Sie jetzt zu Tab 'SOLL Skills zu Jobs' um den Kompetenzabgleich durchzuführen!")
                                                else:
                                                    st.warning(f"Rolle zugewiesen, aber Speichern fehlgeschlagen!")
                                                
                                                st.rerun()
                                            else:
                                                st.warning("Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
                                    else:
                                        st.write("Keine Skills für diese Rolle gefunden.")
                            else:
                                st.warning(f"❌ Kein ESCO-Code für KldB-Code '{selected_kldb_code}' gefunden.")
            
            # Möglichkeit, eine Rolle für automatische Job-Zuordnung auszuwählen
            st.markdown("---")
            st.subheader("🔍 Automatische Job-Zuordnung testen")
            
            selected_role = st.selectbox(
                "Wählen Sie eine Rolle für den Test:",
                [rolle['name'] for rolle in ist_rollen],
                index=0,
                key="xml_role_test_select"
            )
            
            if st.button("🔍 Job-Zuordnung finden", key="job_zuordnung_test_tab1"):
                with st.spinner("Suche passenden KldB-Code..."):
                    kldb_code, kldb_label = find_kldb_code_for_job_title(
                        selected_role, occupations_df, kldb_esco_df
                    )
                
                if kldb_code and kldb_label:
                    st.success(f"✅ Job-Zuordnung gefunden!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Rollenname:** {selected_role}")
                        st.write(f"**KldB-Code:** {kldb_code}")
                    with col2:
                        st.write(f"**KldB-Rolle:** {kldb_label}")
                        
                        # Zeige zugehörige Skills
                        if kldb_code:
                            kldb_match = kldb_esco_df[kldb_esco_df['KldB_Code'] == kldb_code]
                            if not kldb_match.empty:
                                esco_uri = kldb_match.iloc[0].get('ESCO_Code', '')
                                if esco_uri:
                                    role_skills = occupation_skill_relations_df[
                                        occupation_skill_relations_df['occupationUri'] == esco_uri
                                    ]
                                    if not role_skills.empty:
                                        st.write(f"**Verfügbare Skills:** {len(role_skills)}")
                else:
                    st.warning("❌ Keine passende Job-Zuordnung gefunden.")
                    st.info("Tipp: Überprüfen Sie die Schreibweise oder fügen Sie manuell einen KldB-Code hinzu.")
        else:
            st.warning("Keine IST-Rollen in der XML gefunden.")
    
    with tab2:
        st.subheader("🎯 SOLL-Skills zu Jobs - Kompetenzabgleich & Kursempfehlungen")
        
        if soll_skills:
            st.write(f"**Zukünftig benötigte Skills (aus Capabilities):** {len(soll_skills)}")
            
            # Prüfe ob eine IST-Rolle aus Tab 1 ausgewählt wurde
            if ('xml_match_select' not in st.session_state or 
                not st.session_state.get('xml_match_select') or 
                st.session_state.xml_match_select == "Bitte wählen Sie eine gematchte IST-Rolle...") and \
               ('selected_kldb_code_tab2' not in st.session_state or 
                'selected_esco_role_tab2' not in st.session_state):
                
                st.info("💡 **Bitte wählen Sie zuerst eine IST-Rolle im Tab 'IST-Rollen aus XML' aus, um den Kompetenzabgleich durchzuführen.**")
                st.write("**Verfügbare SOLL-Skills (Capabilities):**")
                
                # Tabelle mit SOLL-Skills
                soll_data = []
                for skill in soll_skills:
                    soll_data.append({
                        'Capability': skill['name'],
                        'Typ': skill['type'],
                        'ID': skill.get('id', skill.get('identifier', 'N/A'))
                    })
                
                soll_df = pd.DataFrame(soll_data)
                st.dataframe(soll_df, use_container_width=True)
                return
            
            # Hole die ausgewählte IST-Rolle aus Tab 1 oder der manuellen Auswahl
            selected_match = st.session_state.get('xml_match_select', '')
            manual_selection = st.session_state.get('manual_selection_tab1', '')
            selected_kldb_code = None
            selected_esco_role = None
            
            # Prüfe ob eine manuelle Auswahl aus Tab 1 vorhanden ist
            if 'selected_kldb_code_tab2' in st.session_state and 'selected_esco_role_tab2' in st.session_state:
                selected_kldb_code = st.session_state.selected_kldb_code_tab2
                selected_esco_role = st.session_state.selected_esco_role_tab2
                st.success(f"✅ **Ausgewählte IST-Rolle:** {st.session_state.get('selected_role_data_name', 'Manuell ausgewählte Rolle')} → {selected_esco_role} ({selected_kldb_code})")
            elif manual_selection:
                # Verwende die manuelle Auswahl aus Tab 1
                selected_kldb_code = manual_selection.split("(")[1].split(")")[0]
                selected_esco_role = manual_selection.split(" → ")[1].split(" (")[0] if " → " in manual_selection else ""
                st.success(f"✅ **Ausgewählte IST-Rolle:** {manual_selection}")
            elif selected_match and selected_match != "Bitte wählen Sie eine gematchte IST-Rolle...":
                # Extrahiere KldB-Code aus der ursprünglichen Auswahl
                selected_kldb_code = selected_match.split("(")[1].split(")")[0]
                selected_esco_role = selected_match.split(" → ")[1].split(" (")[0] if " → " in selected_match else ""
                st.success(f"✅ **Ausgewählte IST-Rolle:** {selected_match}")
            else:
                st.warning("❌ Keine IST-Rolle ausgewählt. Bitte wechseln Sie zu Tab 1 und wählen Sie eine Rolle aus.")
                return
            
            if selected_kldb_code:
                # Erstelle das aktuelle Mitarbeiterprofil basierend auf der ausgewählten IST-Rolle
                if 'current_employee_id' in st.session_state:
                    employee_id = st.session_state.current_employee_id
                    current_employee_data = st.session_state.employees_data[st.session_state.employees_data['Employee_ID'] == employee_id].iloc[0]
                    
                    # Verwende die ausgewählte KldB-Rolle und ESCO-Rolle
                    current_kldb = selected_kldb_code
                    current_esco_role = selected_esco_role or current_employee_data.get('ESCO_Role', '')
                    current_manual_skills = current_employee_data.get('Manual_Skills', '')
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
                        current_esco_role,
                        current_manual_essential_skills,
                        current_manual_optional_skills,
                        current_removed_skills
                    )
                    
                    if current_profile:
                        st.markdown("---")
                        st.subheader("📊 Kompetenzabgleich: IST vs. SOLL")
                        
                        # Aktuelle Skills der IST-Rolle
                        current_skills = current_profile['skills']
                        current_skill_labels = [skill['skill_label'].lower() for skill in current_skills]
                        
                        # SOLL-Skills (Capabilities) - Verwende die gleiche Datenquelle wie in der Strategischen Weiterbildung
                        future_skills_from_archi = extract_future_skills_from_capabilities(st.session_state.archi_data)
                        soll_skill_names = [skill['skill_name'].lower() for skill in future_skills_from_archi]
                        
                        # Berechne Übereinstimmungen und fehlende Skills
                        matching_skills = []
                        missing_skills = []
                        
                        for soll_skill in future_skills_from_archi:
                            soll_name = soll_skill['skill_name'].lower()
                            if soll_name in current_skill_labels:
                                # Finde den entsprechenden aktuellen Skill
                                current_skill = next((s for s in current_skills if s['skill_label'].lower() == soll_name), None)
                                if current_skill:
                                    matching_skills.append({
                                        'skill_name': soll_skill['skill_name'],
                                        'current_skill': current_skill,
                                        'is_essential': current_skill.get('is_essential', False)
                                    })
                            else:
                                missing_skills.append(soll_skill)
                        
                        # NEUE SEKTION: Zugewiesene Skills anzeigen
                        st.markdown("---")
                        st.subheader("🔧 Zugewiesene Skills des aktuellen Mitarbeiters")
                        
                        if current_skills:
                            st.success(f"**Anzahl zugewiesener Skills:** {len(current_skills)}")
                            
                            # Gruppiere Skills nach Typ
                            essential_skills = [s for s in current_skills if s.get('is_essential', False)]
                            optional_skills = [s for s in current_skills if not s.get('is_essential', False)]
                            manual_skills = [s for s in current_skills if s.get('relation_type') in ['manual', 'manual_essential', 'manual_optional']]
                            automatic_skills = [s for s in current_skills if s.get('relation_type') not in ['manual', 'manual_essential', 'manual_optional']]
                            
                            # Zeige Skills in Spalten
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**🔥 Essential Skills:**")
                                if essential_skills:
                                    for skill in essential_skills:
                                        skill_type = ""
                                        if skill.get('relation_type') == 'manual_essential':
                                            skill_type = " (manuell hinzugefügt)"
                                        elif skill.get('relation_type') == 'manual':
                                            skill_type = " (manuell)"
                                        else:
                                            skill_type = " (automatisch)"
                                        st.write(f"• {skill['skill_label']}{skill_type}")
                                else:
                                    st.write("Keine Essential Skills zugewiesen")
                                
                                st.write("**💡 Optional Skills:**")
                                if optional_skills:
                                    for skill in optional_skills:
                                        skill_type = ""
                                        if skill.get('relation_type') == 'manual_optional':
                                            skill_type = " (manuell hinzugefügt)"
                                        elif skill.get('relation_type') == 'manual':
                                            skill_type = " (manuell)"
                                        else:
                                            skill_type = " (automatisch)"
                                        st.write(f"• {skill['skill_label']}{skill_type}")
                                else:
                                    st.write("Keine Optional Skills zugewiesen")
                            
                            with col2:
                                st.write("**📊 Skill-Statistiken:**")
                                st.write(f"• **Essential Skills:** {len(essential_skills)}")
                                st.write(f"• **Optional Skills:** {len(optional_skills)}")
                                st.write(f"• **Manuelle Skills:** {len(manual_skills)}")
                                st.write(f"• **Automatische Skills:** {len(automatic_skills)}")
                                
                                # Zeige aktuelle Rolle
                                if current_profile.get('current_role'):
                                    current_role = current_profile['current_role']
                                    st.write("**👤 Aktuelle Rolle:**")
                                    st.write(f"• **KldB-Code:** {current_role.get('KldB_Code', 'N/A')}")
                                    st.write(f"• **KldB-Rolle:** {current_role.get('KldB_Label', 'N/A')}")
                                    st.write(f"• **ESCO-Rolle:** {current_role.get('ESCO_Label', 'N/A')}")
                        else:
                            st.warning("**Keine Skills zugewiesen.** Bitte weisen Sie zuerst eine Rolle in 'Mitarbeiter-Kompetenzprofile' zu.")
                        

                        
                        # Berechne Prozentsatz der Übereinstimmung
                        match_percentage = (len(matching_skills) / len(future_skills_from_archi)) * 100 if future_skills_from_archi else 0
                        

                        
                        # NEUE SEKTION: Zukünftig benötigte Skills (aus Capabilities) - 1:1 wie in Strategische Weiterbildung
                        st.markdown("---")
                        st.subheader("Zukünftig benötigte Skills (aus Capabilities)")
                        
                        # Verwende die gleiche Datenquelle wie in der Strategischen Weiterbildung
                        future_skills_from_archi = extract_future_skills_from_capabilities(st.session_state.archi_data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Zukünftig benötigte Skills (aus Capabilities):**")
                            st.write(f"• Anzahl: {len(future_skills_from_archi)}")
                            
                            # Zeige alle zukünftigen Skills
                            for skill in future_skills_from_archi[:10]:  # Zeige nur die ersten 10
                                st.write(f"  - {skill['skill_name']}")
                            if len(future_skills_from_archi) > 10:
                                st.write(f"  ... und {len(future_skills_from_archi) - 10} weitere")
                        
                        with col2:
                            st.write("**✅ Übereinstimmende Skills:**")
                            if matching_skills:
                                st.success(f"{len(matching_skills)} von {len(future_skills_from_archi)} SOLL-Skills bereits vorhanden")
                                
                                # Gruppiere nach Essential/Optional
                                essential_matches = [s for s in matching_skills if s['is_essential']]
                                optional_matches = [s for s in matching_skills if not s['is_essential']]
                                
                                if essential_matches:
                                    st.write("**🔥 Essential Skills:**")
                                    for skill in essential_matches:
                                        st.write(f"• {skill['skill_name']}")
                                
                                if optional_matches:
                                    st.write("**💡 Optional Skills:**")
                                    for skill in optional_matches:
                                        st.write(f"• {skill['skill_name']}")
                            else:
                                st.warning("Keine Übereinstimmungen gefunden")
                        
                        # Kursempfehlungen für fehlende Skills
                        if missing_skills:
                            st.markdown("---")
                            st.subheader("📚 Kursempfehlungen für fehlende Skills")
                            
                            st.info(f"🎯 **Generiere Kursempfehlungen für {len(missing_skills)} fehlende Skills...**")
                            
                            if st.button("🎓 Kursempfehlungen generieren", key="generate_course_recommendations_tab2"):
                                with st.spinner("Generiere Kursempfehlungen für fehlende Skills..."):
                                    try:
                                        # Konvertiere fehlende Skills in das richtige Format
                                        missing_skill_names = [skill['skill_name'] for skill in missing_skills]
                                        
                                        # Begrenze die Anzahl der Skills für bessere Performance
                                        max_skills_to_process = min(10, len(missing_skill_names))
                                        st.info(f"⚡ Verarbeite die ersten {max_skills_to_process} fehlenden Skills für bessere Performance")
                                        
                                        # Rufe die Funktion auf
                                        recommendations = find_udemy_courses_for_skills(
                                            missing_skill_names[:max_skills_to_process],
                                            udemy_courses_df,
                                            top_k=3
                                        )
                                        
                                        if recommendations:
                                            st.success(f"✅ {len(recommendations)} Kursempfehlungen gefunden!")
                                            
                                            # Gruppiere nach Skill
                                            skill_groups = {}
                                            for rec in recommendations:
                                                skill = rec.get('skill', 'Unbekannt')
                                                if skill not in skill_groups:
                                                    skill_groups[skill] = []
                                                skill_groups[skill].append(rec)
                                            
                                            # Zeige Empfehlungen gruppiert nach Skill mit Scoring
                                            for skill_name, skill_recs in skill_groups.items():
                                                st.write(f"**📖 Kurse für: {skill_name}**")
                                                
                                                # Sortiere nach Similarity Score
                                                skill_recs.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                                                
                                                for i, course in enumerate(skill_recs, 1):
                                                    with st.expander(f"🎯 {i}. {course.get('course_title', 'Unbekannter Kurs')} - Score: {course.get('similarity_score', 0):.3f}", expanded=False):
                                                        col1, col2, col3 = st.columns([3, 1, 1])
                                                        
                                                        with col1:
                                                            st.write(f"**Kursname:** {course.get('course_title', 'N/A')}")
                                                            if course.get('course_headline') and course.get('course_headline') != 'N/A':
                                                                st.write(f"**Beschreibung:** {course.get('course_headline')}")
                                                            if course.get('course_description') and course.get('course_description') != 'N/A':
                                                                desc = course.get('course_description')
                                                                if len(str(desc)) > 200:
                                                                    desc = str(desc)[:200] + "..."
                                                                st.write(f"**Details:** {desc}")
                                                        
                                                        with col2:
                                                            if course.get('course_price') and course.get('course_price') != 'N/A':
                                                                st.write(f"**Preis:** {course.get('course_price')}")
                                                            if course.get('course_language') and course.get('course_language') != 'N/A':
                                                                st.write(f"**Sprache:** {course.get('course_language')}")
                                                        
                                                        with col3:
                                                            if course.get('course_url'):
                                                                st.write(f"**Link:** [Zum Kurs]({course.get('course_url')})")
                                                            if course.get('similarity_score'):
                                                                # Farbkodierung basierend auf Score
                                                                score = course.get('similarity_score', 0)
                                                                if score >= 0.8:
                                                                    st.success(f"**Match-Score:** {score:.3f} ⭐⭐⭐")
                                                                elif score >= 0.6:
                                                                    st.info(f"**Match-Score:** {score:.3f} ⭐⭐")
                                                                else:
                                                                    st.warning(f"**Match-Score:** {score:.3f} ⭐")
                                                
                                                st.markdown("---")
                                        else:
                                            st.warning("❌ Keine Kursempfehlungen gefunden!")
                                            st.info("💡 Versuchen Sie es mit anderen Skills oder überprüfen Sie die Udemy-Daten.")
                                            
                                            # Fallback: Zeige manuelle Beispielkurse
                                            st.subheader("📚 Beispiel-Kursempfehlungen")
                                            st.info("Da keine automatischen Empfehlungen gefunden wurden, zeigen wir Ihnen allgemeine Kursempfehlungen:")
                                            
                                            for skill_name in missing_skill_names[:5]:
                                                st.write(f"**Für Skill: {skill_name}**")
                                                st.write("- Grundlagen und Einführungskurse")
                                                st.write("- Fortgeschrittene Techniken")
                                                st.write("- Praktische Anwendungen")
                                                st.write("- Zertifizierungskurse")
                                                st.markdown("---")
                                    
                                    except Exception as e:
                                        st.error(f"Fehler bei der Kursempfehlung: {str(e)}")
                                        st.info("💡 Bitte versuchen Sie es erneut oder kontaktieren Sie den Administrator.")
                                        st.exception(e)
                        else:
                            st.success("🎉 Alle SOLL-Skills sind bereits vorhanden! Keine Kursempfehlungen nötig.")
                    
                    else:
                        st.error("❌ Konnte kein Mitarbeiterprofil für die ausgewählte IST-Rolle erstellen.")
                else:
                    st.warning("⚠️ Bitte wählen Sie zuerst einen Mitarbeiter in der Sidebar aus.")
        else:
            st.warning("Keine SOLL-Skills in der XML gefunden.")
    


if __name__ == "__main__":
    main()