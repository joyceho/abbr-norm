'''
Scrape Nursing Labs Abbreviations webpage for common nursing abbreviations
'''    
import scrapy


class NurseLabsAbbrScraper(scrapy.Spider):
    name = "nurseAbbr"
    start_urls = [
        'https://nurseslabs.com/medical-terminologies-abbreviations-listcheat-sheet/',
    ]

    def parse(self, response):
        for row in response.xpath('//table//tr[position()>1]'):
            yield {
                'abbr': row.xpath('td[1]//text()').extract(),
                'full': row.xpath('td[2]//text()').extract(),
            }