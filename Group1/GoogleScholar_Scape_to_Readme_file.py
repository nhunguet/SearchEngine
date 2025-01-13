import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

def scrape_google_scholar(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  
    soup = BeautifulSoup(response.text, 'html.parser')

    papers = []  # List to store each paper's data

    entries = soup.find_all('tr', class_='gsc_a_tr')  # Assuming each paper is within a table row

    for entry in entries:
        title_tag = entry.find('a', class_='gsc_a_at')
        citation_tag = entry.find('a', class_='gsc_a_ac gs_ibl')
        author_tag = entry.find('div', class_='gs_gray')
        year_tag = entry.find('span', class_='gsc_a_h gsc_a_hc gs_ibl')

        paper = {
            "Title": title_tag.text if title_tag else "No title",
            "URL": 'https://scholar.google.com' + title_tag['href'] if title_tag else "No URL",
            "Citations": citation_tag.text if citation_tag else " ",
            "Authors": author_tag.text if author_tag else "Unknown",
            "Year": year_tag.text if year_tag else "N/A"
        }
        papers.append(paper)

    return papers


def write_to_readme(papers):

    current_datetime = datetime.now(timezone.utc)
    current_date_time_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Tạo chuỗi HTML từ dữ liệu và thêm thông tin thời gian
    html_content = "\n\n<table id=\"scholar-table\" style=\"position: relative;\">\n"
    html_content += "  <tr>\n"
    html_content += "    <th>Title</th>\n"
    html_content += "    <th>Authors</th>\n"
    html_content += "    <th>Citations</th>\n"
    html_content += "    <th>Year</th>\n"
    html_content += "  </tr>\n"

    for paper in papers:
        html_content += (
            f"  <tr>\n     <td><a href=\"{paper['URL']}\">{paper['Title']}</a></td>\n"
            f"    <td>{paper['Authors']}</td>\n    <td>{paper['Citations']}</td>\n"
            f"    <td>{paper['Year']}</td>\n  </tr>\n"
        )

    # Add the "Show more" row with center alignment, larger font size, and italicized text
    # html_content += f"  <tr>\n    <td align=\"center\"   colspan=\"4\" id=\"show-more-cell\" "
    # html_content += f"style=\"text-align:center; font-size: larger; position: relative;\" "
    # html_content += f"title=\"Last Updated: {current_date_time_str}\">\n"
    # html_content += f"<em><a href=\"{data['user_scholar_url']}\" style=\"display: inline-block;\">Show more</a></em></td>\n  </tr>\n</table>\n\n"

    html_content += "</table>\n\n"
    html_content += f"<p>Last Updated: {current_date_time_str}</p>"

    # Đọc toàn bộ README.md
    with open("README.md", "r", encoding="utf-8") as readme_file:
        readme_content = readme_file.read()

    # Tìm vị trí bắt đầu và kết thúc của phần cần thay thế
    start_marker = "<!-- SCHOLAR-LIST:START -->"
    end_marker = "<!-- SCHOLAR-LIST:END -->"
    start_pos = readme_content.find(start_marker) + len(start_marker)
    end_pos = readme_content.find(end_marker)

    # Thay thế phần giữa start_pos và end_pos bằng nội dung mới của bảng và thông tin thời gian
    new_readme_content = (
        readme_content[:start_pos] + html_content + readme_content[end_pos:]
    )

    # Ghi nội dung mới vào README.md
    with open("README.md", "w", encoding="utf-8") as readme_file:
        readme_file.write(new_readme_content)


url = 'https://scholar.google.com/citations?user=ztzAuOMAAAAJ&hl=en' # my scholar
papers = scrape_google_scholar(url)
write_to_readme(papers)


