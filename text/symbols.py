'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
20180126-add support for arpabet
'''
from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from .korean import ALL_SYMBOLS, PAD, EOS
from text import cmudict

# For english
#en_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '+EOS+PAD #<- For LJSpeech_1_0_2018-01-04_23-19-23 EOS location 63(enter it into train.py input_length calc part)

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

en_symbols = [PAD,EOS]+list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ')+_arpabet  #<-For deployment(Because korean ALL_SYMBOLS follow this convention)

symbols = list(ALL_SYMBOLS) # for korean
