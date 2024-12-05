import arxivscraper
import datetime
import time
import requests
import json
from datetime import timedelta
import os
import pathlib


def get_daily_code(DateToday,cats):
    """
    @param DateToday: str
    @param cats: dict
    @return paper_with_code: dict
    """
    from_day = until_day = DateToday
    content = dict()
    # content
    output = dict()
    for k,v in cats.items():
        scraper = arxivscraper.Scraper(category=k, date_from=from_day,date_until=until_day,filters={'categories':v})
        tmp = scraper.scrape()
        print(tmp)
        if isinstance(tmp,list):
            for item in tmp:
                if item["id"] not in output:
                    output[item["id"]] = item
        time.sleep(30)

    base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
    cnt = 0

    for k,v in output.items():
        print(v["id"])
        _id = v["id"]
        paper_title = " ".join(v["title"].split())
        paper_url = v["url"]
        url = base_url + _id
        try:
            r = requests.get(url).json()
            if "official" in r and r["official"]:
                cnt += 1
                repo_url = r["official"]["url"]
                repo_name = repo_url.split("/")[-1]

                content[_id] = f"|[{paper_title}]({paper_url})|[{repo_name}]({repo_url})|\n"
        except Exception as e:
            print(f"exception: {e} with id: {_id}")
    data = {DateToday:content}
    return data

def update_daily_json(filename,data_all):
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)
    
    #将datas更新到m中
    for data in data_all:
        m.update(data)

    # save data to daily.json

    with open(filename,"w") as f:
        json.dump(m,f)
    



def ensure_archive_dirs():
    """Create archive directory structure if it doesn't exist"""
    archive_base = "archives"
    current_year = datetime.date.today().year
    
    # Create base archives dir
    pathlib.Path(archive_base).mkdir(exist_ok=True)
    
    # Create directories for all years from 2020 to current year
    for year in range(2020, current_year + 1):
        year_dir = os.path.join(archive_base, str(year))
        pathlib.Path(year_dir).mkdir(exist_ok=True)
        
        # Create month directories if they don't exist
        for month in range(1, 13):
            month_file = os.path.join(year_dir, f"{month:02d}.md")
            if not os.path.exists(month_file):
                with open(month_file, "w") as f:
                    f.write(f"# {datetime.date(year, month, 1).strftime('%B %Y')} Archive\n\n")

def json_to_md(filename):
    """
    Convert JSON data to markdown files with archives
    @param filename: str
    @return None
    """
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)
    
    # Ensure archive structure exists
    ensure_archive_dirs()
    
    # Group entries by year and month
    entries_by_month = {}
    latest_entries = []
    today = datetime.date.today()
    week_ago = today - timedelta(days=7)
    
    for day in sorted(data.keys(), reverse=True):
        day_date = datetime.datetime.strptime(day, "%Y-%m-%d").date()
        year_month = day_date.strftime("%Y/%m")
        
        # Collect entries for archives
        if day_date > week_ago:
            latest_entries.append((day, data[day]))
            
        if year_month not in entries_by_month:
            entries_by_month[year_month] = []
        entries_by_month[year_month].append((day, data[day]))
    
    # Update main README.md
    with open("README.md", "w") as f:
        # Write header and overview
        f.write("# Daily ArXiv\n\n")
        f.write("HelloGitHub shares interesting, entry-level GitHub projects. Updated monthly on the 28th in the form of a monthly magazine. ")
        f.write("Content includes: interesting and entry-level projects, open source books, practical projects, enterprise projects, etc., ")
        f.write("allowing you to experience the charm of open source in a very short time and fall in love with open source!\n\n")
        
        f.write("## Overview\n")
        f.write("This project automatically analyzes papers from specific categories on arXiv daily using GitHub Actions, ")
        f.write("identifying papers that have released their source code and appending them to this repository.\n\n")
        
        # Write latest updates
        f.write("## Latest Updates (Last 7 Days)\n")
        for day, day_content in latest_entries:
            if not day_content:
                continue
            f.write(f"### {day}\n")
            f.write("|paper|code|\n" + "|---|---|\n")
            for k, v in day_content.items():
                f.write(v)
            f.write("\n")
        
        # Write archive links
        f.write("\n## Archives\n")
        for year_month in sorted(entries_by_month.keys(), reverse=True):
            year, month = year_month.split("/")
            month_name = datetime.date(int(year), int(month), 1).strftime("%B")
            f.write(f"- [{month_name} {year}](archives/{year}/{int(month):02d}.md)\n")
    
    # Update archive files
    for year_month, entries in entries_by_month.items():
        year, month = year_month.split("/")
        archive_file = f"archives/{year}/{int(month):02d}.md"
        
        with open(archive_file, "w") as f:
            f.write(f"# {datetime.date(int(year), int(month), 1).strftime('%B %Y')} Archive\n\n")
            for day, day_content in entries:
                if not day_content:
                    continue
                f.write(f"## {day}\n")
                f.write("|paper|code|\n" + "|---|---|\n")
                for k, v in day_content.items():
                    f.write(v)
                f.write("\n")
    
    print("Finished generating markdown files")

if __name__ == "__main__":

    DateToday = datetime.date.today()
    N = 5 # 往前查询的天数
    data_all = []
    for i in range(1,N):
        day = str(DateToday + timedelta(-i))
        # you can add the categories in cats
        cats = {
        "eess":["eess.SP"],
        "cs":["cs.IT"]
    }
        data = get_daily_code(day,cats)
        data_all.append(data)
    update_daily_json("daily.json",data_all)
    json_to_md("daily.json")
