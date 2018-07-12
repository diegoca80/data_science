import argparse
import json
from collections import OrderedDict
from datetime import datetime
import random
__author__ = 'diego.alves'
 
class Fig(list):
    def __init__(self, values):
        self.values = values
        with open("brazil-cities-states.json", encoding="utf-8") as statefile:
            self.state_city_dict = json.load(statefile, object_pairs_hook=OrderedDict)
            self.state = "" 
            self.city = ""
        with open("names.json", encoding="utf-8") as namefile:
            self.name_list = json.load(namefile, object_pairs_hook=OrderedDict)
        
        
    def generator(self):
        for field in self.values:
            if field[1] == 'DATE':
                date = self.getDate()
                print(date + self.getBlank(field[0], len(date)), end='')
            elif field[1] == 'BRANCOS':
                print(self.getBlank(field[0]), end='')
            elif field[1] == 'CPF_CNPJ':
                cnpj = self.getCNPJ()
                print(cnpj + self.getBlank(field[0], len(cnpj)), end='')
            elif field[1] == 'NUMERIC':
                numeric = self.getNumeric(field[0])
                print(numeric + self.getBlank(field[0], len(numeric)), end='')
            elif field[1] == 'STATUS':
                print('AT', end='')
            elif field[1] == 'DOMAIN':
                domain = self.getDomain(field[2])
                print(domain + self.getBlank(field[0], len(domain)), end='')
            elif field[1] == 'STATE':
                self.state = self.getState()
                print(self.state + self.getBlank(field[0], len(self.state)), end='')
            elif field[1] == 'CITY':
                if self.state == "":
                    self.state = self.getState()
                self.city = self.getCity(self.state)
                print(self.city + self.getBlank(field[0], len(self.city)), end ='')
            elif field[1] == 'NAME':
                self.name = self.getName()
                print(self.name + self.getBlank(field[0], len(self.name)), end ='')
        return ''
    
    def getCPF(self):
        cpf = [random.randint(0, 9) for x in range(9)]                                                                                          
        for _ in range(2):                             
            val = sum([(len(cpf) + 1 - i) * v for i, v in enumerate(cpf)]) % 11 
            cpf.append(11 - val if val > 1 else 0)     
        return '%s%s%s%s%s%s%s%s%s%s%s' % tuple(cpf)
    
    def getCNPJ(self):
        cnpj =  [1, 0, 0, 0] + [random.randint(0, 9) for x in range(8)]
        for _ in range(2):                                                          
            digit = 0
            for i, v in enumerate(cnpj):
                digit += v * (i % 8 + 2)                                     
            digit = 11 - digit % 11
            cnpj = [digit if digit < 10 else 0] + cnpj                                                                   
        return '%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % tuple(cnpj[::-1])

    def getDate(self, start_year = 2015, final_year = 2017):
        year = random.randint(start_year, final_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        valid_date = datetime(year, month, day)
        return valid_date.strftime("%d%m%Y")
    
    def getDomain(self, values):
        return random.choice(values)
    
    def getNumeric(self, length):
        range_end = (10**length)-1
        return '%s' % random.choice(range(0, range_end))
    
    def getState(self):
        return self.state_city_dict["estados"][random.randint(0,26)]["nome"]
        
    def getCity(self, state):
        for i in range(len(self.state_city_dict["estados"])):
            if self.state_city_dict["estados"][i]["nome"] == self.state:
                return self.state_city_dict["estados"][i]["cidades"][random.randint(0, len(self.state_city_dict["estados"][i]["cidades"]))]

    def getName(self):
        return self.name_list[random.randint(0, len(self.name_list))]
    
    def getBlank(self, field_length, filled = 0):
        return '' if filled >= field_length else ' '*(field_length-filled)

def main(input, output, num_records):
	with open(args.input) as jsonfile:
		layout_dict = json.load(jsonfile, object_pairs_hook=OrderedDict)
	print('Generating sample ' + layout_dict["layout"] + '\n')
	count = int(args.num_records)
	while(count > 0):
		for record_type in layout_dict["record_types"]:
			for record in layout_dict["record_types"][record_type]:
				print(record_type, end='')
				fig = Fig(record.values())
				print(fig.generator())
				count -= 1
	print('\nGenerated with success: %s' % args.output)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This is a data generator for enriched files.')
	parser.add_argument('-i','--input', help='Input file name',required=True)
	parser.add_argument('-o','--output',help='Output file name', required=True)
	parser.add_argument('-n','--num_records',help='Number of records', required=True)
	args = parser.parse_args()
	main(args.input, args.output, args.num_records)