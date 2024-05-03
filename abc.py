Data_to_run = []
for filename in os.listdir('txt/'):
        if filename[:4] in ['C1_4', 'C2_4', 'R1_4', 'R2_4'] or filename[:5] in ['RC1_4', 'RC2_4']:
            Data_to_run.append(filename)

print(Data_to_run)