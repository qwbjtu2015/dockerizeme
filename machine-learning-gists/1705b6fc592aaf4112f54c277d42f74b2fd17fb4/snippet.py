import win32com.client # sender and subject of the email

outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

inbox = outlook.GetDefaultFolder(6) # "6" refers to the index of a folder - in this case,
                                    # the inbox. You can change that number to reference
                                    # any other folder
messages = inbox.Items

messages.Sort("ReceivedTime", True)

key = 'Machine Learning'

print("Total Inbox Mails = {0}".format(len(messages)))

total = 0


from sets import Set
unique_senders = Set([])


for i, message in enumerate(messages):
    try: 
        sender =  message.SenderName
        subject = message.subject
        if key in subject:        
            total = total + 1
            print (total, ": ", sender.encode('utf-8'), "--> ", subject.encode('utf-8'))
            unique_senders.add(sender)
    except:
        pass

print("Total Attention Mails = {0}".format(total))
print("Total Unique Senders = {0}".format(len(unique_senders)))

f = open('unique_senders.txt', 'w+')
       
for item in list(unique_senders):
    f.write("%s\n" % item)

f.close()
    
#message = messages.GetLast()
#body_content = message.body
#print body_content