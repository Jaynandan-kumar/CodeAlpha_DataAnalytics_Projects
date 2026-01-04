# 🎬 IMDb Top Movies Web Scraping Project  
### CodeAlpha Data Analytics Internship – Task 1

---

## 📋 Project Overview
This project is my **submission for Task 1: Web Scraping** of the **CodeAlpha Data Analytics Internship**.  
The goal of this project is to demonstrate **real-world web scraping skills** by extracting movie data from the **IMDb Top 250 Movies** webpage using Python and saving it into a structured CSV file.

The script is designed to be **robust, ethical, and beginner-friendly**, with proper error handling and fallback mechanisms to ensure data availability even if live scraping fails.

---

## 🎯 Objectives
- Extract real-world data from the IMDb website using Python  
- Parse and analyze HTML structure using BeautifulSoup  
- Convert unstructured web data into a clean, structured dataset  
- Save extracted data into a CSV file for further analysis  
- Implement error handling and fallback sample data  
- Follow ethical web scraping practices (rate limiting & headers)

---

## 🌍 Real-World Use Cases
This type of web scraping project can be used in:
- 🎥 Movie recommendation systems  
- 📊 Entertainment industry data analysis  
- 📈 Trend analysis of popular movies over time  
- 🧠 Machine Learning datasets for rating prediction  
- 📚 Academic and research projects  

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Requests** – Fetching web pages  
- **BeautifulSoup4** – HTML parsing  
- **Pandas** – Data manipulation & analysis  
- **CSV Module** – Data export  
- **Time Module** – Rate limiting for ethical scraping  

---

## 📊 Data Collected
Each movie record contains the following fields:

| Column Name | Description |
|------------|------------|
| `rank` | Position in IMDb Top list |
| `title` | Movie title |
| `year` | Release year |
| `rating` | IMDb rating (out of 10) |
| `director` | Movie director |

---

## ✨ Features
- ✅ Scrapes IMDb Top Movies list  
- ✅ Extracts title, year, rating, and director  
- ✅ Limits scraping to first 20–30 movies (ethical scraping)  
- ✅ Graceful error handling for network or structure changes  
- ✅ Automatic fallback to sample dataset if scraping fails  
- ✅ Saves data in CSV format  
- ✅ Displays summary statistics in console  

---

## 📂 Project Structure
CodeAlpha_WebScraping/
│
├── web_scraping_imdb.py
├── imdb_movies.csv
└── README.md
