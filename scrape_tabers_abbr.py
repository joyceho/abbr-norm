import scrapy
import re

class TabersSpider(scrapy.Spider):
    name = 'tabers'
    start_urls = ['https://www.tabers.com/tabersonline/view/Tabers-Dictionary/767492/all/Medical_Abbreviations']

    def parse(self, response):
        pClass = response.xpath('//section[@class="section"]/p')
        bText = pClass[0].xpath('b')
        for i in range(len(bText)):
            abbrText = bText[i].xpath('text()').extract_first().strip()
            fullText = bText[i].xpath('following-sibling::text()').extract_first().strip()
            # if there are multiple meanings or empty string, skip it
            if ';' in fullText or fullText == '':
                continue
            yield {
                'abbr': [abbrText],
                'full': [fullText]
            }