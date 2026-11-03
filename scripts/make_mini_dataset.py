#!/usr/bin/env python3
"""
Dataset Generator for Miniature Offline Smoke Testing (Milestone M1).
Generates authentic SQLite databases (financial_mini, california_schools),
sample BIRD benchmark queries, and synthetic reasoning trajectories (mini_synsql).
"""

import os
import sys
import sqlite3
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Dict, Any, Optional
from src.grounding.column_card import ColumnCardExtractor
from src.grounding.graph_builder import SchemaGraphBuilder


DATA_MINI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mini")


def create_financial_db(db_path: str) -> None:
    """Creates financial_mini.sqlite with 5 related tables and multi-hop PK-FK relations."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE districts (
        district_id INTEGER PRIMARY KEY,
        district_name TEXT NOT NULL,
        region TEXT NOT NULL,
        population INTEGER NOT NULL
    );

    CREATE TABLE clients (
        client_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        gender TEXT NOT NULL,
        birth_date TEXT NOT NULL,
        district_id INTEGER NOT NULL,
        FOREIGN KEY (district_id) REFERENCES districts(district_id)
    );

    CREATE TABLE accounts (
        account_id INTEGER PRIMARY KEY,
        client_id INTEGER NOT NULL,
        frequency TEXT NOT NULL,
        creation_date TEXT NOT NULL,
        FOREIGN KEY (client_id) REFERENCES clients(client_id)
    );

    CREATE TABLE cards (
        card_id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL,
        card_type TEXT NOT NULL,
        issued_date TEXT NOT NULL,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    );

    CREATE TABLE transactions (
        trans_id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL,
        trans_date TEXT NOT NULL,
        trans_type TEXT NOT NULL,
        amount REAL NOT NULL,
        balance REAL NOT NULL,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    );
    """)

    # Populate districts
    districts_data = [
        (1, "Prague-Central", "Bohemia", 1200000),
        (2, "Brno-City", "Moravia", 380000),
        (3, "Ostrava-North", "Silesia", 290000),
        (4, "Plzen-West", "Bohemia", 175000),
    ]
    cur.executemany("INSERT INTO districts VALUES (?, ?, ?, ?);", districts_data)

    # Populate clients
    clients_data = [
        (101, "Jan Novak", "M", "1975-04-12", 1),
        (102, "Marie Dvorakova", "F", "1982-11-23", 1),
        (103, "Petr Cech", "M", "1989-07-08", 2),
        (104, "Eva Svobodova", "F", "1994-01-30", 3),
        (105, "Tomas Rosicky", "M", "1980-10-04", 4),
        (106, "Lucie Bila", "F", "1966-04-07", 2),
        (107, "Pavel Nedved", "M", "1972-08-30", 1),
        (108, "Klara Koukalova", "F", "1982-02-24", 4),
    ]
    cur.executemany("INSERT INTO clients VALUES (?, ?, ?, ?, ?);", clients_data)

    # Populate accounts
    accounts_data = [
        (1001, 101, "monthly", "2020-01-15"),
        (1002, 102, "monthly", "2019-05-20"),
        (1003, 103, "weekly", "2021-03-10"),
        (1004, 104, "monthly", "2021-08-14"),
        (1005, 105, "monthly", "2018-12-01"),
        (1006, 106, "weekly", "2022-02-18"),
        (1007, 107, "monthly", "2017-09-05"),
        (1008, 108, "monthly", "2023-04-11"),
    ]
    cur.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?);", accounts_data)

    # Populate cards (Notice: linked to accounts, not clients directly!)
    cards_data = [
        (5001, 1001, "classic", "2020-02-01"),
        (5002, 1002, "gold", "2019-06-15"),
        (5003, 1003, "junior", "2021-04-01"),
        (5004, 1005, "gold", "2019-01-10"),
        (5005, 1007, "platinum", "2017-10-01"),
        (5006, 1008, "classic", "2023-05-02"),
    ]
    cur.executemany("INSERT INTO cards VALUES (?, ?, ?, ?);", cards_data)

    # Populate transactions
    trans_data = [
        (8001, 1001, "2023-01-05", "credit", 15000.0, 15000.0),
        (8002, 1001, "2023-01-10", "withdrawal", 2500.0, 12500.0),
        (8003, 1002, "2023-01-08", "credit", 45000.0, 45000.0),
        (8004, 1002, "2023-01-15", "withdrawal", 12000.0, 33000.0),
        (8005, 1003, "2023-01-20", "credit", 8000.0, 8000.0),
        (8006, 1004, "2023-01-22", "credit", 22000.0, 22000.0),
        (8007, 1005, "2023-01-12", "credit", 60000.0, 60000.0),
        (8008, 1005, "2023-01-25", "withdrawal", 15000.0, 45000.0),
        (8009, 1006, "2023-02-01", "credit", 14000.0, 14000.0),
        (8010, 1007, "2023-02-05", "credit", 95000.0, 95000.0),
        (8011, 1007, "2023-02-18", "withdrawal", 20000.0, 75000.0),
        (8012, 1008, "2023-04-15", "credit", 18000.0, 18000.0),
    ]
    cur.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?);", trans_data)

    conn.commit()
    conn.close()


def create_schools_db(db_path: str) -> None:
    """Creates california_schools.sqlite with 3 related tables (schools, frpm, satscores)."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE schools (
        CDSCode TEXT PRIMARY KEY,
        School TEXT NOT NULL,
        District TEXT NOT NULL,
        County TEXT NOT NULL,
        Charter INTEGER NOT NULL
    );

    CREATE TABLE frpm (
        CDSCode TEXT PRIMARY KEY,
        Free_Meal_Count REAL,
        Percent_Eligible REAL,
        Charter_School_Status TEXT,
        FOREIGN KEY (CDSCode) REFERENCES schools(CDSCode)
    );

    CREATE TABLE satscores (
        cds TEXT PRIMARY KEY,
        sname TEXT,
        dname TEXT,
        NumTstTakr INTEGER,
        AvgScrMath INTEGER,
        AvgScrRead INTEGER,
        AvgScrWrit INTEGER,
        NumGE1500 INTEGER,
        FOREIGN KEY (cds) REFERENCES schools(CDSCode)
    );
    """)

    schools_data = [
        ("01611760130062", "Albany High", "Albany City Unified", "Alameda", 0),
        ("01611760130070", "MacGregor High", "Albany City Unified", "Alameda", 0),
        ("01612590131581", "Berkeley High", "Berkeley Unified", "Alameda", 0),
        ("19647331930775", "Belmont Senior High", "Los Angeles Unified", "Los Angeles", 0),
        ("19647331932623", "Fairfax Senior High", "Los Angeles Unified", "Los Angeles", 0),
        ("19647331933852", "Granada Hills Charter", "Los Angeles Unified", "Los Angeles", 1),
        ("38684783830205", "Mission High", "San Francisco Unified", "San Francisco", 0),
        ("38684783830403", "Lowell High", "San Francisco Unified", "San Francisco", 0),
        ("43694844330031", "Cupertino High", "Fremont Union High", "Santa Clara", 0),
        ("43694844334314", "Lynbrook High", "Fremont Union High", "Santa Clara", 0),
    ]
    cur.executemany("INSERT INTO schools VALUES (?, ?, ?, ?, ?);", schools_data)

    frpm_data = [
        ("01611760130062", 312.0, 24.5, "No"),
        ("01611760130070", 45.0, 68.2, "No"),
        ("01612590131581", 950.0, 32.1, "No"),
        ("19647331930775", 820.0, 88.4, "No"),
        ("19647331932623", 1120.0, 71.0, "No"),
        ("19647331933852", 1540.0, 48.6, "Yes"),
        ("38684783830205", 780.0, 75.3, "No"),
        ("38684783830403", 420.0, 16.8, "No"),
        ("43694844330031", 180.0, 9.2, "No"),
        ("43694844334314", 110.0, 6.1, "No"),
    ]
    cur.executemany("INSERT INTO frpm VALUES (?, ?, ?, ?);", frpm_data)

    sat_data = [
        ("01611760130062", "Albany High", "Albany City Unified", 285, 625, 605, 595, 210),
        ("01611760130070", "MacGregor High", "Albany City Unified", 18, 410, 420, 390, 2),
        ("01612590131581", "Berkeley High", "Berkeley Unified", 560, 590, 580, 565, 340),
        ("19647331930775", "Belmont Senior High", "Los Angeles Unified", 240, 430, 415, 410, 35),
        ("19647331932623", "Fairfax Senior High", "Los Angeles Unified", 310, 490, 480, 470, 95),
        ("19647331933852", "Granada Hills Charter", "Los Angeles Unified", 780, 585, 570, 560, 490),
        ("38684783830205", "Mission High", "San Francisco Unified", 195, 460, 450, 440, 45),
        ("38684783830403", "Lowell High", "San Francisco Unified", 650, 680, 660, 650, 580),
        ("43694844330031", "Cupertino High", "Fremont Union High", 460, 690, 655, 660, 410),
        ("43694844334314", "Lynbrook High", "Fremont Union High", 490, 715, 670, 680, 450),
    ]
    cur.executemany("INSERT INTO satscores VALUES (?, ?, ?, ?, ?, ?, ?, ?);", sat_data)

    conn.commit()
    conn.close()


def generate_mini_bird_jsonl(output_path: str, fin_db_path: str, sch_db_path: str) -> None:
    """Generates 24 sample BIRD benchmark queries with difficulty levels and verified SQL."""
    queries = [
        # financial_mini queries (1-12)
        {
            "question_id": 1,
            "db_id": "financial_mini",
            "question": "How many clients are registered in the Bohemia region?",
            "evidence": "Bohemia region is stored in the districts table.",
            "SQL": "SELECT COUNT(T1.client_id) FROM clients AS T1 JOIN districts AS T2 ON T1.district_id = T2.district_id WHERE T2.region = 'Bohemia';",
            "difficulty": "simple",
        },
        {
            "question_id": 2,
            "db_id": "financial_mini",
            "question": "What is the average transaction amount for account 1001?",
            "evidence": "Average amount can be computed using AVG(amount).",
            "SQL": "SELECT AVG(amount) FROM transactions WHERE account_id = 1001;",
            "difficulty": "simple",
        },
        {
            "question_id": 3,
            "db_id": "financial_mini",
            "question": "List the names of all clients who own a gold card.",
            "evidence": "Clients connect to accounts, which connect to cards. card_type = 'gold'.",
            "SQL": "SELECT T1.name FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN cards AS T3 ON T2.account_id = T3.account_id WHERE T3.card_type = 'gold';",
            "difficulty": "challenging", # Multi-hop bridge table test
        },
        {
            "question_id": 4,
            "db_id": "financial_mini",
            "question": "Find the total balance of all accounts belonging to male clients.",
            "evidence": "Male clients have gender = 'M'. Balance is in transactions. Latest balance or total transactions.",
            "SQL": "SELECT SUM(T3.balance) FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN transactions AS T3 ON T2.account_id = T3.account_id WHERE T1.gender = 'M';",
            "difficulty": "challenging",
        },
        {
            "question_id": 5,
            "db_id": "financial_mini",
            "question": "Which account had the highest transaction amount?",
            "evidence": "Sort by amount descending and limit 1.",
            "SQL": "SELECT account_id FROM transactions ORDER BY amount DESC LIMIT 1;",
            "difficulty": "simple",
        },
        {
            "question_id": 6,
            "db_id": "financial_mini",
            "question": "What are the names of clients living in Brno-City?",
            "evidence": "District name is 'Brno-City'.",
            "SQL": "SELECT T1.name FROM clients AS T1 JOIN districts AS T2 ON T1.district_id = T2.district_id WHERE T2.district_name = 'Brno-City';",
            "difficulty": "moderate",
        },
        {
            "question_id": 7,
            "db_id": "financial_mini",
            "question": "How many cards have been issued for weekly accounts?",
            "evidence": "Weekly accounts have frequency = 'weekly'.",
            "SQL": "SELECT COUNT(T1.card_id) FROM cards AS T1 JOIN accounts AS T2 ON T1.account_id = T2.account_id WHERE T2.frequency = 'weekly';",
            "difficulty": "moderate",
        },
        {
            "question_id": 8,
            "db_id": "financial_mini",
            "question": "What is the maximum balance recorded across all transactions in accounts owned by Marie Dvorakova?",
            "evidence": "Marie Dvorakova is in clients table.",
            "SQL": "SELECT MAX(T3.balance) FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN transactions AS T3 ON T2.account_id = T3.account_id WHERE T1.name = 'Marie Dvorakova';",
            "difficulty": "challenging",
        },
        {
            "question_id": 9,
            "db_id": "financial_mini",
            "question": "Count the number of withdrawal transactions.",
            "evidence": "trans_type = 'withdrawal'.",
            "SQL": "SELECT COUNT(*) FROM transactions WHERE trans_type = 'withdrawal';",
            "difficulty": "simple",
        },
        {
            "question_id": 10,
            "db_id": "financial_mini",
            "question": "Find the card type and issued date for client Jan Novak.",
            "evidence": "Multi-hop join from clients through accounts to cards.",
            "SQL": "SELECT T3.card_type, T3.issued_date FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN cards AS T3 ON T2.account_id = T3.account_id WHERE T1.name = 'Jan Novak';",
            "difficulty": "challenging",
        },
        {
            "question_id": 11,
            "db_id": "financial_mini",
            "question": "List the distinct frequencies of accounts created by clients in district 1.",
            "evidence": "district_id = 1.",
            "SQL": "SELECT DISTINCT T2.frequency FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id WHERE T1.district_id = 1;",
            "difficulty": "moderate",
        },
        {
            "question_id": 12,
            "db_id": "financial_mini",
            "question": "Which client owns the platinum card?",
            "evidence": "card_type = 'platinum'.",
            "SQL": "SELECT T1.name FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN cards AS T3 ON T2.account_id = T3.account_id WHERE T3.card_type = 'platinum';",
            "difficulty": "challenging",
        },

        # california_schools queries (13-24)
        {
            "question_id": 13,
            "db_id": "california_schools",
            "question": "What is the average math SAT score for schools in Santa Clara County?",
            "evidence": "County is in schools table, AvgScrMath is in satscores table.",
            "SQL": "SELECT AVG(T2.AvgScrMath) FROM schools AS T1 JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.County = 'Santa Clara';",
            "difficulty": "moderate",
        },
        {
            "question_id": 14,
            "db_id": "california_schools",
            "question": "Find the names of schools where more than 50% of students are eligible for free meals.",
            "evidence": "Percent_Eligible > 50 in frpm table.",
            "SQL": "SELECT T1.School FROM schools AS T1 JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.Percent_Eligible > 50.0;",
            "difficulty": "moderate",
        },
        {
            "question_id": 15,
            "db_id": "california_schools",
            "question": "What is the reading SAT score for the school with the highest free meal count?",
            "evidence": "Schools connects frpm and satscores. Free_Meal_Count in frpm, AvgScrRead in satscores.",
            "SQL": "SELECT T3.AvgScrRead FROM frpm AS T1 JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode JOIN satscores AS T3 ON T2.CDSCode = T3.cds ORDER BY T1.Free_Meal_Count DESC LIMIT 1;",
            "difficulty": "challenging", # Multi-hop bridge table test (schools bridges frpm and satscores)
        },
        {
            "question_id": 16,
            "db_id": "california_schools",
            "question": "How many schools are charter schools in Los Angeles County?",
            "evidence": "Charter = 1 and County = 'Los Angeles'.",
            "SQL": "SELECT COUNT(*) FROM schools WHERE Charter = 1 AND County = 'Los Angeles';",
            "difficulty": "simple",
        },
        {
            "question_id": 17,
            "db_id": "california_schools",
            "question": "List the top 3 schools with the highest writing SAT scores.",
            "evidence": "Order by AvgScrWrit descending limit 3.",
            "SQL": "SELECT sname FROM satscores ORDER BY AvgScrWrit DESC LIMIT 3;",
            "difficulty": "simple",
        },
        {
            "question_id": 18,
            "db_id": "california_schools",
            "question": "What is the total number of free meals provided in Alameda County schools?",
            "evidence": "Sum Free_Meal_Count where County = 'Alameda'.",
            "SQL": "SELECT SUM(T2.Free_Meal_Count) FROM schools AS T1 JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.County = 'Alameda';",
            "difficulty": "moderate",
        },
        {
            "question_id": 19,
            "db_id": "california_schools",
            "question": "Which school has an average math SAT score above 700?",
            "evidence": "AvgScrMath > 700.",
            "SQL": "SELECT sname FROM satscores WHERE AvgScrMath > 700;",
            "difficulty": "simple",
        },
        {
            "question_id": 20,
            "db_id": "california_schools",
            "question": "Find the percent of eligible students for free meals in Lowell High.",
            "evidence": "School name is 'Lowell High'.",
            "SQL": "SELECT T2.Percent_Eligible FROM schools AS T1 JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.School = 'Lowell High';",
            "difficulty": "moderate",
        },
        {
            "question_id": 21,
            "db_id": "california_schools",
            "question": "What is the average math SAT score for charter schools?",
            "evidence": "Charter schools have Charter = 1 in schools table.",
            "SQL": "SELECT AVG(T2.AvgScrMath) FROM schools AS T1 JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.Charter = 1;",
            "difficulty": "moderate",
        },
        {
            "question_id": 22,
            "db_id": "california_schools",
            "question": "For schools where free meal eligibility is less than 20 percent, what is the average math score?",
            "evidence": "Connect frpm to schools to satscores. Percent_Eligible < 20.",
            "SQL": "SELECT AVG(T3.AvgScrMath) FROM frpm AS T1 JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode JOIN satscores AS T3 ON T2.CDSCode = T3.cds WHERE T1.Percent_Eligible < 20.0;",
            "difficulty": "challenging",
        },
        {
            "question_id": 23,
            "db_id": "california_schools",
            "question": "How many test takers took the SAT in Berkeley High?",
            "evidence": "sname = 'Berkeley High'.",
            "SQL": "SELECT NumTstTakr FROM satscores WHERE sname = 'Berkeley High';",
            "difficulty": "simple",
        },
        {
            "question_id": 24,
            "db_id": "california_schools",
            "question": "List all districts in Los Angeles County that have schools.",
            "evidence": "Distinct District from schools where County = 'Los Angeles'.",
            "SQL": "SELECT DISTINCT District FROM schools WHERE County = 'Los Angeles';",
            "difficulty": "simple",
        },
    ]

    # Verify each query executes and produces valid results
    conn_fin = sqlite3.connect(fin_db_path)
    conn_sch = sqlite3.connect(sch_db_path)

    for q in queries:
        conn = conn_fin if q["db_id"] == "financial_mini" else conn_sch
        try:
            cur = conn.cursor()
            cur.execute(q["SQL"])
            res = cur.fetchall()
            assert len(res) > 0, f"Query {q['question_id']} returned empty result: {q['SQL']}"
        except Exception as e:
            raise RuntimeError(f"Error validating query {q['question_id']}: {q['SQL']}. Error: {e}")

    conn_fin.close()
    conn_sch.close()

    # Write queries to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def generate_mini_synsql_jsonl(output_path: str) -> None:
    """Generates synthetic reasoning trajectories with <think> CoT and <answer> SQL."""
    samples = [
        {
            "id": "syn_01",
            "db_id": "financial_mini",
            "question": "List the names of clients who have a gold card.",
            "schema": "CREATE TABLE clients (client_id INT PRIMARY KEY, name TEXT);\nCREATE TABLE accounts (account_id INT PRIMARY KEY, client_id INT, FOREIGN KEY (client_id) REFERENCES clients(client_id));\nCREATE TABLE cards (card_id INT PRIMARY KEY, account_id INT, card_type TEXT, FOREIGN KEY (account_id) REFERENCES accounts(account_id));",
            "think_steps": (
                "1. User wants names of clients with gold cards.\n"
                "2. Table 'clients' has the 'name' column.\n"
                "3. Table 'cards' contains 'card_type', with condition card_type = 'gold'.\n"
                "4. Note that 'clients' and 'cards' do not share a direct foreign key. They are linked via bridge table 'accounts'.\n"
                "5. JOIN clients AS T1 ON T1.client_id = T2.client_id with accounts AS T2.\n"
                "6. JOIN accounts AS T2 ON T2.account_id = T3.account_id with cards AS T3.\n"
                "7. Filter with WHERE T3.card_type = 'gold' and SELECT T1.name."
            ),
            "answer": "SELECT T1.name FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN cards AS T3 ON T2.account_id = T3.account_id WHERE T3.card_type = 'gold';",
        },
        {
            "id": "syn_02",
            "db_id": "financial_mini",
            "question": "What is the total transaction amount for account 1001?",
            "schema": "CREATE TABLE transactions (trans_id INT PRIMARY KEY, account_id INT, amount REAL);",
            "think_steps": (
                "1. User wants total transaction amount for a specific account.\n"
                "2. 'transactions' table contains account_id and amount.\n"
                "3. Use aggregate function SUM(amount).\n"
                "4. Filter condition WHERE account_id = 1001."
            ),
            "answer": "SELECT SUM(amount) FROM transactions WHERE account_id = 1001;",
        },
        {
            "id": "syn_03",
            "db_id": "california_schools",
            "question": "What is the reading score of the school with the highest free meal count?",
            "schema": "CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, School TEXT);\nCREATE TABLE frpm (CDSCode TEXT PRIMARY KEY, Free_Meal_Count REAL, FOREIGN KEY (CDSCode) REFERENCES schools(CDSCode));\nCREATE TABLE satscores (cds TEXT PRIMARY KEY, AvgScrRead INT, FOREIGN KEY (cds) REFERENCES schools(CDSCode));",
            "think_steps": (
                "1. Question asks for reading score (AvgScrRead in satscores).\n"
                "2. Condition specifies the school with the highest free meal count (Free_Meal_Count in frpm).\n"
                "3. 'frpm' and 'satscores' connect through bridge table 'schools' on CDSCode.\n"
                "4. Order by T1.Free_Meal_Count DESC and LIMIT 1.\n"
                "5. SELECT T3.AvgScrRead."
            ),
            "answer": "SELECT T3.AvgScrRead FROM frpm AS T1 JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode JOIN satscores AS T3 ON T2.CDSCode = T3.cds ORDER BY T1.Free_Meal_Count DESC LIMIT 1;",
        },
        {
            "id": "syn_04",
            "db_id": "california_schools",
            "question": "How many schools are in Los Angeles County?",
            "schema": "CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, County TEXT);",
            "think_steps": (
                "1. We need the count of schools.\n"
                "2. Filter by County = 'Los Angeles'.\n"
                "3. Use COUNT(*) on schools table."
            ),
            "answer": "SELECT COUNT(*) FROM schools WHERE County = 'Los Angeles';",
        },
        {
            "id": "syn_05",
            "db_id": "financial_mini",
            "question": "Find the maximum balance of client Jan Novak.",
            "schema": "CREATE TABLE clients (client_id INT PRIMARY KEY, name TEXT);\nCREATE TABLE accounts (account_id INT PRIMARY KEY, client_id INT, FOREIGN KEY (client_id) REFERENCES clients(client_id));\nCREATE TABLE transactions (trans_id INT PRIMARY KEY, account_id INT, balance REAL, FOREIGN KEY (account_id) REFERENCES accounts(account_id));",
            "think_steps": (
                "1. Target metric: maximum balance, represented by MAX(T3.balance).\n"
                "2. Client name is 'Jan Novak' in clients table.\n"
                "3. Join clients to accounts, then accounts to transactions.\n"
                "4. WHERE T1.name = 'Jan Novak'."
            ),
            "answer": "SELECT MAX(T3.balance) FROM clients AS T1 JOIN accounts AS T2 ON T1.client_id = T2.client_id JOIN transactions AS T3 ON T2.account_id = T3.account_id WHERE T1.name = 'Jan Novak';",
        },
    ]

    # Generate additional synthetic variations to reach 20 samples
    for i in range(6, 21):
        if i % 2 == 0:
            samples.append({
                "id": f"syn_{i:02d}",
                "db_id": "financial_mini",
                "question": f"List client names from district {i % 4 + 1}.",
                "schema": "CREATE TABLE clients (client_id INT PRIMARY KEY, name TEXT, district_id INT);",
                "think_steps": (
                    f"1. Query asks for client names.\n"
                    f"2. Filter by district_id = {i % 4 + 1}.\n"
                    f"3. SELECT name FROM clients WHERE district_id = {i % 4 + 1};"
                ),
                "answer": f"SELECT name FROM clients WHERE district_id = {i % 4 + 1};",
            })
        else:
            samples.append({
                "id": f"syn_{i:02d}",
                "db_id": "california_schools",
                "question": f"What is the average math score for schools in county Alameda with charter status {i % 2}?",
                "schema": "CREATE TABLE schools (CDSCode TEXT PRIMARY KEY, County TEXT, Charter INT);\nCREATE TABLE satscores (cds TEXT PRIMARY KEY, AvgScrMath INT, FOREIGN KEY (cds) REFERENCES schools(CDSCode));",
                "think_steps": (
                    "1. Need AVG(AvgScrMath) from satscores table.\n"
                    f"2. Join with schools on CDSCode.\n"
                    f"3. Filter on County = 'Alameda' AND Charter = {i % 2}."
                ),
                "answer": f"SELECT AVG(T2.AvgScrMath) FROM schools AS T1 JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.County = 'Alameda' AND T1.Charter = {i % 2};",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            # Add formatted prompt matching Completion-Only loss training specification
            s["formatted_prompt"] = (
                f"<prompt>\nQuestion: {s['question']}\nSchema:\n{s['schema']}\n</prompt>\n"
                f"<think>\n{s['think_steps']}\n</think>\n"
                f"<answer>\n{s['answer']}\n</answer>"
            )
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def generate_all(output_dir: Optional[str] = None) -> Dict[str, str]:
    """Generates all miniature dataset assets into target directory."""
    target_dir = os.path.abspath(output_dir or DATA_MINI_DIR)
    os.makedirs(target_dir, exist_ok=True)

    fin_db_path = os.path.join(target_dir, "financial_mini.sqlite")
    sch_db_path = os.path.join(target_dir, "california_schools.sqlite")
    bird_path = os.path.join(target_dir, "mini_bird.jsonl")
    synsql_path = os.path.join(target_dir, "mini_synsql.jsonl")

    create_financial_db(fin_db_path)
    create_schools_db(sch_db_path)
    generate_mini_bird_jsonl(bird_path, fin_db_path, sch_db_path)
    generate_mini_synsql_jsonl(synsql_path)

    cards_path = os.path.join(target_dir, "column_cards_mini.jsonl")
    extractor = ColumnCardExtractor(sample_limit=3)
    cards_fin = extractor.extract_from_sqlite(fin_db_path)
    cards_sch = extractor.extract_from_sqlite(sch_db_path)
    all_cards = cards_fin + cards_sch
    ColumnCardExtractor.save_cards_to_jsonl(all_cards, cards_path)

    graph_fin = SchemaGraphBuilder.build_from_sqlite(fin_db_path)
    graph_sch = SchemaGraphBuilder.build_from_sqlite(sch_db_path)
    graphs_dict = {"financial_mini": graph_fin, "california_schools": graph_sch}
    graphs_path = os.path.join(target_dir, "graphs_mini.pkl")
    import pickle
    with open(graphs_path, "wb") as f:
        pickle.dump(graphs_dict, f)

    return {
        "financial_mini": fin_db_path,
        "california_schools": sch_db_path,
        "mini_bird": bird_path,
        "mini_synsql": synsql_path,
        "column_cards": cards_path,
        "graphs": graphs_path,
    }


def main():
    print(f"Creating mini dataset directory at: {DATA_MINI_DIR}")
    res = generate_all(DATA_MINI_DIR)
    print("\n✅ Miniature dataset creation complete!")
    for k, v in res.items():
        print(f"  - {v} (exists: {os.path.exists(v)})")


if __name__ == "__main__":
    main()
