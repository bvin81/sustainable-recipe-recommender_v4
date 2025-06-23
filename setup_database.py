#!/usr/bin/env python3
"""
Egyszerűsített setup - CSV feldolgozás és adatbázis inicializálás
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def setup_project():
    """Teljes projekt inicializálás"""
    print("🚀 Project Setup - Simplified Version")
    print("=" * 50)
    
    # Könyvtárak létrehozása
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    user_study_dir = Path("user_study")
    user_study_dir.mkdir(exist_ok=True)
    
    # __init__.py fájlok létrehozása
    create_init_files()
    
    # CSV feldolgozás
    success = setup_csv()
    
    if success:
        print("\n🎉 PROJECT SETUP COMPLETE!")
        print("✅ CSV processed successfully")
        print("✅ Directory structure created")
        print("✅ Ready for deployment")
    else:
        print("\n⚠️ SETUP COMPLETED WITH FALLBACKS")
        print("✅ App will work with sample data")
    
    return success

def create_init_files():
    """__init__.py fájlok létrehozása"""
    
    # user_study/__init__.py
    init_content = '''"""
User Study Module - Simplified Version
"""
from .routes import user_study_bp

__all__ = ['user_study_bp']
'''
    
    with open("user_study/__init__.py", "w", encoding="utf-8") as f:
        f.write(init_content)
    
    print("✅ __init__.py files created")

def setup_csv():
    """CSV feldolgozás egyszerűsítve"""
    
    original_csv = Path("hungarian_recipes_github.csv")
    output_csv = Path("data/processed_recipes.csv")
    
    print(f"📊 Looking for: {original_csv}")
    
    # Ha van eredeti CSV, feldolgozzuk
    if original_csv.exists():
        try:
            print("📋 Processing original CSV...")
            
            # Több encoding próbálása
            df = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(original_csv, encoding=encoding)
                    print(f"✅ Loaded with {encoding} encoding")
                    break
                except:
                    continue
            
            if df is None:
                raise Exception("Could not load with any encoding")
            
            # Egyszerű feldolgozás
            processed_df = process_csv(df)
            processed_df.to_csv(output_csv, index=False, encoding='utf-8')
            
            print(f"✅ Processed CSV saved: {len(processed_df)} recipes")
            return True
            
        except Exception as e:
            print(f"⚠️ CSV processing error: {e}")
    
    # Fallback: sample adatok
    print("🔧 Creating sample recipes...")
    create_sample_csv(output_csv)
    return False

def process_csv(df):
    """CSV feldolgozás egyszerűsítve"""
    
    print(f"📊 Processing {len(df)} recipes")
    
    # Oszlop normalizálás
    column_mapping = {
        'id': 'recipeid',
        'name': 'title', 
        'recipe_name': 'title',
        'image': 'images',
        'image_url': 'images',
        'directions': 'instructions',
        'steps': 'instructions'
    }
    
    # Átnevezés
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Kötelező oszlopok biztosítása
    required_columns = ['recipeid', 'title', 'ingredients', 'instructions', 'images']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'recipeid':
                df[col] = range(1, len(df) + 1)
            elif col == 'images':
                df[col] = generate_image_urls(len(df))
            else:
                df[col] = f'Default {col}'
    
    # Tisztítás
    df = df.fillna('')
    df = df.drop_duplicates(subset=['title'], keep='first')
    df = df.reset_index(drop=True)
    df['recipeid'] = range(1, len(df) + 1)
    
    # Pontszámok hozzáadása
    add_scores(df)
    
    print(f"✅ Processed to {len(df)} clean recipes")
    return df

def create_sample_csv(output_csv):
    """Sample CSV generálása"""
    
    # Alapreceptek
    base_recipes = [
        ("Gulyásleves", "marhahús, hagyma, paprika, paradicsom, burgonya", "Levesek"),
        ("Vegetáriánus Lecsó", "paprika, paradicsom, hagyma, tojás", "Vegetáriánus"),
        ("Halászlé", "ponty, csuka, hagyma, paradicsom, paprika", "Halételek"),
        ("Túrós Csusza", "széles metélt, túró, tejföl, szalonna", "Tésztaételek"),
        ("Gombapaprikás", "gomba, hagyma, paprika, tejföl", "Vegetáriánus"),
        ("Schnitzel", "sertéshús, liszt, tojás, zsemlemorzsa", "Húsételek"),
        ("Töltött Káposzta", "savanyú káposzta, darált hús, rizs", "Húsételek"),
        ("Rántott Sajt", "trappista sajt, liszt, tojás", "Vegetáriánus"),
        ("Babgulyás", "bab, hagyma, paprika, kolbász", "Levesek"),
        ("Palócleves", "bárány, bab, burgonya, tejföl", "Levesek")
    ]
    
    # 50 receptre bővítés
    sample_data = []
    for i in range(50):
        base = base_recipes[i % len(base_recipes)]
        recipe_id = i + 1
        title = base[0] + (f" - {i//len(base_recipes) + 1}. változat" if i >= len(base_recipes) else "")
        
        sample_data.append({
            'recipeid': recipe_id,
            'title': title,
            'ingredients': base[1],
            'instructions': f"Főzési útmutató a {title} recepthez.",
            'category': base[2],
            'images': f'https://images.unsplash.com/photo-154759218{i%10}-recipe?w=400&h=300'
        })
    
    df = pd.DataFrame(sample_data)
    add_scores(df)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✅ Sample CSV created: {len(df)} recipes")

def add_scores(df):
    """Pontszámok hozzáadása"""
    np.random.seed(42)  # Reprodukálható eredmények
    
    n = len(df)
    df['ESI'] = np.clip(np.random.normal(65, 15, n), 10, 100)  # Környezeti
    df['HSI'] = np.clip(np.random.normal(70, 12, n), 20, 100)  # Egészség
    df['PPI'] = np.clip(np.random.normal(75, 10, n), 30, 100)  # Népszerűség
    
    # Összetett pontszám
    df['composite_score'] = (df['ESI'] * 0.4 + df['HSI'] * 0.4 + df['PPI'] * 0.2)
    
    # Kerekítés
    for col in ['ESI', 'HSI', 'PPI', 'composite_score']:
        df[col] = df[col].round(2)

def generate_image_urls(count):
    """Placeholder képek generálása"""
    base_urls = [
        'https://images.unsplash.com/photo-1547592180-85f173990554?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1544943910-4c1dc44aab44?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1565299507177-b0ac66763828?w=400&h=300&fit=crop'
    ]
    
    return [base_urls[i % len(base_urls)] for i in range(count)]

if __name__ == "__main__":
    setup_project()
