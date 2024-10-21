from bs4 import BeautifulSoup
import requests
import re
from PyPDF2 import PdfReader, PdfWriter
import pdfkit

PDFSEARCHQUERY = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?query=site%3Ancbi.nlm.nih.gov+" \
        "{0}&v%3Aproject=nlm-main-website&v:state=root%7Croot-{1}-50%7C0&"
PDFQUERY = "https://www.ncbi.nlm.nih.gov"
PERPAGE = 50
PAGEUNAVAILABLE = "did not return any results"
PDFUNAVAILABLE = "Page not available"
HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def ScrapperPrompt():
    """Prompt user for search string and max items and return the values"""
    print("Search Term:", end=" ")
    searchString = input()

    maxSearchValue = ""
    while not maxSearchValue.isdigit():
        # Loops until user inputs a valid number
        print("Maximum Number of Files:", end=" ")
        maxSearchValue = input()

        if not maxSearchValue.isdigit():
            print("Please input a valid number!")

    return searchString, maxSearchValue


def CleanName(fileName):
    """Clean the file name to remove invalid characters"""
    return re.sub(r'[\\/*?:"<>|.]', "", fileName)


def AppendSourceMetadata(fileName, source):
    """Append the source to PDF metadata"""
    try:
        reader = PdfReader("Resources/" + fileName)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        metadata = reader.metadata
        writer.add_metadata(metadata)
        writer.add_metadata({"/Source": source})

        with open("Resources/" + fileName, "wb") as f:
            writer.write(f)
            f.close()

        print("Successfully appended metadata to: " + fileName)
    except:
        print("Error in appending metadata!")


def ScrapePDF():
    """Scrape the NIH website for PDFs"""
    searchString, maxSearchValue = ScrapperPrompt()
    pageCount = 0
    searchString = searchString.strip().replace(" ", "+")
    maxSearchValue = int(maxSearchValue)

    while True:
        # Create the search query based on page index and get search string
        searchQuery = PDFSEARCHQUERY.format(searchString, pageCount * PERPAGE)
        urlData = requests.get(searchQuery).text

        if PAGEUNAVAILABLE in urlData:
            # If page is unavailable or has no more pages, exit loop
            break
        else:
            # If page exists, get webpages URLs and save valid PDFs
            urlist = re.findall(r"<span class=\"url\">(https://www.ncbi.nlm.nih.gov/books/.*?)</span>", urlData)

            for url in urlist:
                maxSearchValue -= DownloadPDF(url)

                if maxSearchValue <= 0:
                    break

        pageCount += 1

        if maxSearchValue <= 0:
            break


def DownloadPDF(url):
    """If URL is valid, download the PDF or scrape the Print View and save it as a PDF"""
    print("Searching using URL: " + url)
    urlData = requests.get(url, headers=HEADER).text
    urlList = re.findall(r'<li><a href="([^"]*?)">PDF version of this title</a>', urlData)
    titleChunk = re.findall(r'<div class="icnblk_cntnt(?: [^"]*)?">.*?<h2>(.*?)<\/h2>|<span itemprop="name">(.*?)<\/span>', urlData, re.DOTALL)
    
    if urlList:
        # Attempt to save PDF if it exists
        pdfURL = PDFQUERY + urlList[0]
        pdfData = requests.get(pdfURL, headers=HEADER)
        code = url.split("/")[-1] + ".pdf"

        if PDFUNAVAILABLE in pdfData.text:
            # If the PDF is unavailable, return 0
            print("Unavailable PDF")
            return 0
        else:
            # If PDF is available, save PDF
            urlName = code

            if titleChunk is None:
                # If no title, use code as file name
                pass
            elif not titleChunk:
                # If fail to match, use code as file name
                pass
            else:
                # If chunk is found, attempt to extract title
                for match in titleChunk:
                    title = match[0] if match[0] else match[1]

                    if title:
                        urlName = CleanName(title) + ".pdf"
            
            try:
                # Save PDF
                with open("Resources/" + urlName, "wb") as f:
                    f.write(pdfData.content)

                f.close()
                print("Successfully Created: " + urlName)

                AppendSourceMetadata(urlName, "PubMed Books")

                return 1
            except:
                print("Something went wrong in saving the PDF!")
                return 0
    else:
        # No PDF version for the title, check if Print View is available
        print("No PDF version for this title!")
        printViewList = re.findall(r'<a href="([^"]*?)">Print View</a>', urlData)

        if printViewList:
            printViewURL = PDFQUERY + printViewList[0]
            print("Scraping Print View: " + printViewURL)
            printViewData = requests.get(printViewURL, headers=HEADER).text

            # Instead of scraping HTML, pass the full URL to pdfkit.from_url
            try:
                # Create a filename for the PDF
                code = url.split("/")[-1]
                urlName = CleanName(code) + ".pdf"

                # Use pdfkit.from_url to save the Print View page as a PDF with external resources
                pdfkit.from_url(printViewURL, "Resources/" + urlName)

                print("Successfully Created: " + urlName)

                AppendSourceMetadata(urlName, "PubMed Books")

                return 1
            except Exception as e:
                print(f"Something went wrong in saving the Print View as PDF: {e}")
                return 0
        else:
            print("No Print View available for this title!")
            return 0