"""
An english to pig latin translator made without the use of machine learning.
"""
s = input('Enter a string to translate to Pig Latin: ')

end = ''
vowels = ['a', 'e', 'i', 'o', 'u']
n = 0

for i in range(len(s)):
    if s[i] in vowels:
        n = i
        break
    else:
        end += s[i]

print(s[n:]+end+'ay')
