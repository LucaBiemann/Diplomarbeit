from random import randrange
import pandas as pd


def ausweisnummern(n):
    zeichen = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'C', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T',
               'V', 'W', 'X', 'Y', 'Z']
    start = ['L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y']
    gewichtung = [7, 3, 1,7, 3, 1,7, 3, 1]
    buchstaben = pd.DataFrame(columns=['buchstabe', 'wert'])
    buchstaben['buchstabe'] = ['C', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'V', 'W', 'X', 'Y', 'Z']
    buchstaben['wert'] = [12,15,16,17,19,20,21,22,23,25,27,29,31,32,33,34,35]
    ausweis_list = []
    for i in range(n):
        ausweisnr = start[randrange(10)] + zeichen[randrange(27)] + zeichen[randrange(27)] + zeichen[randrange(27)] + zeichen[
            randrange(27)] + zeichen[randrange(27)] + zeichen[randrange(27)] + zeichen[randrange(27)] + zeichen[randrange(27)]
        pruefsumme = 0
        for i, o in zip(ausweisnr,gewichtung):

            if i in list(buchstaben['buchstabe']):
                ziffer = buchstaben.loc[buchstaben.buchstabe == i, 'wert'].values
                ziffer = ziffer[0]
            else:
                ziffer = int(i)
            produkt = o * ziffer
            endziffer = str(produkt)[-1]
            pruefsumme += int(endziffer)
        pruefziffer = str(pruefsumme)[-1]
        ausweisnr = ausweisnr + pruefziffer
        ausweis_list.append(ausweisnr)
    return ausweis_list
