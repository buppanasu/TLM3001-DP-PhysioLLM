import requests
from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Constants
COCHRANE_LIBRARY_BASE_URL = "https://www.cochranelibrary.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}
START_YEAR = 2003  
CURRENT_YEAR = 2024  
PDF_DOWNLOAD_FOLDER = "pdf_downloads"  


# Selenium WebDriver setup
def init_driver(download_folder):
    """Initialize Selenium WebDriver with specified download folder."""
    options = webdriver.ChromeOptions()
    
    prefs = {
        "download.default_directory": os.path.abspath(download_folder), 
        "download.prompt_for_download": False, 
        "plugins.always_open_pdf_externally": True  
    }
    
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless")  
    driver = webdriver.Chrome(options=options)
    
    return driver



def scrape_issues_and_search(search_term, max_pdfs):
    """Scrape issues from each year and search for reviews matching the search term."""
    reviews = []
    
    for year in range(CURRENT_YEAR, START_YEAR - 1, -1):  
        for issue in range(1, 13): 
            issue_url = f"{COCHRANE_LIBRARY_BASE_URL}/cdsr/table-of-contents/{year}/{issue}"
            print(f"Scraping year {year}, issue {issue}: {issue_url}")

            scrape_reviews_from_issue(issue_url, search_term, max_pdfs, reviews)

            if len(reviews) >= max_pdfs:
                return reviews

    return reviews


def scrape_reviews_from_issue(issue_url, search_term, max_pdfs, reviews):
    """Scrape reviews from a specific issue and match the search term."""
    try:
        response = requests.get(issue_url, headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find reviews in the current issue
        review_elements = soup.select("div.search-results-item-body")

        if not review_elements:
            print(f"No reviews found for issue URL: {issue_url}")
            return

        # Iterate through the reviews and check for matches
        for review_element in review_elements:
            title_element = review_element.select_one("a")
            author_element = review_element.select_one("div.search-result-authors")
            date_element = review_element.select_one("div.search-result-date")

            if title_element:
                title = title_element.get_text(strip=True)
                url = title_element['href'] if title_element['href'].startswith('http') else COCHRANE_LIBRARY_BASE_URL + title_element['href']

                if search_term.lower() in title.lower():
                    review_data = {
                        "title": title,
                        "url": url,
                        "author": author_element.get_text(strip=True) if author_element else "Unknown",
                        "date": date_element.get_text(strip=True) if date_element else "Unknown"
                    }
                    reviews.append(review_data)

                    print(f"Found matching review: {title}, URL: {review_data['url']}")

                    if len(reviews) >= max_pdfs:
                        break

    except requests.RequestException as e:
        print(f"Error scraping reviews from issue {issue_url}: {e}")


def download_pdf(driver, review_url):
    """Access the review page, click the 'Download PDF' button, and download either Full or Summary PDF."""
    driver.get(review_url)

    try:
        # Wait until the PDF dropdown menu is clickable and click it
        wait = WebDriverWait(driver, 10)
        download_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "li.tools.pdf.pulldown-menu")))
        download_button.click()

        # Wait until the dropdown is visible
        time.sleep(2)  # A short sleep to ensure menu visibility, adjust if necessary

        # Check if Full PDF is locked by looking for the "locked" class in the span
        full_pdf_span = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.readcube-label")))

        if "locked" not in full_pdf_span.get_attribute("class"):
            # Full PDF is available, proceed to download
            full_pdf_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pdf-link-full.readcube-epdf-link")))
            full_pdf_button.click()
            print("Accessed PDF link...")

            # Wait for the iframe to appear and switch to it
            iframe = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'iframe.rc-reader-frame'))
            )
            driver.switch_to.frame(iframe)
            print("Switched to iframe")

            # Now inside the iframe, locate and click the pdf button
            try:
                pdf_button = WebDriverWait(driver, 6).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'menu-button.download[data-tippy-content="Save PDF"]'))
                )
                pdf_button.click()
                print("PDF button clicked!")

                # Wait for the download to start (adjust time if necessary)
                time.sleep(5)

                # Save the PDF (You might need to customize this part based on how your browser handles downloads)
                print(f"Full PDF downloaded and saved in folder: {PDF_DOWNLOAD_FOLDER}")

            except Exception as e:
                print(f"Error: {e}")

        else:
            # Full PDF is locked, try to download Summary PDF instead
            print("Full PDF is locked, trying to download Summary PDF...")

            summary_pdf_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pdf-link-abstract.readcube-epdf-link")))
            summary_pdf_button.click()
            print("Accessed PDF link...")

            # Wait for the iframe to appear and switch to it
            iframe = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'iframe.rc-reader-frame'))
            )
            driver.switch_to.frame(iframe)
            print("Switched to iframe")

            # Now inside the iframe, locate and click the pdf button
            try:
                pdf_button = WebDriverWait(driver, 7).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'menu-button.download[data-tippy-content="Save PDF"]'))
                )
                pdf_button.click()
                print("PDF button clicked!")

                # Wait for the download to start (adjust time if necessary)
                time.sleep(7)

                # Save the PDF (You might need to customize this part based on how your browser handles downloads)
                print(f"Abstract PDF downloaded and saved in folder: {PDF_DOWNLOAD_FOLDER}")

            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Error during the PDF download process: {e}")


def create_download_folder(folder_name):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")


def save_reviews_to_file(reviews, file_name="search_results.txt"):
    """Save the scraped reviews to a file."""
    try:
        with open(file_name, 'w') as file:
            for review in reviews:
                file.write(f"{review['title']} | {review['author']} | {review['date']} | {review['url']}\n")
        print(f"Reviews successfully saved to {file_name}")
    except Exception as e:
        print(f"Error saving reviews to file: {e}")


def main():
    """Main function to run the search and download process."""
    search_term = input("Enter search term: ").strip()
    max_pdfs = int(input("Enter the maximum number of PDFs to download: "))

    # Step 1: Scrape issues and search for reviews
    reviews = scrape_issues_and_search(search_term, max_pdfs)

    if not reviews:
        print(f"No reviews found for the search term '{search_term}'.")
        return

    # Step 2: Save reviews to a text file
    save_reviews_to_file(reviews)

    # Step 3: Create folder to save the PDFs
    create_download_folder(PDF_DOWNLOAD_FOLDER)

    # Step 4: Download PDFs for each review using Selenium
    driver = init_driver(PDF_DOWNLOAD_FOLDER)
    for review in reviews:
        print(f"Attempting to download PDF for: {review['title']}")
        download_pdf(driver, review['url'])

    driver.quit()

if __name__ == "__main__":
    main()
