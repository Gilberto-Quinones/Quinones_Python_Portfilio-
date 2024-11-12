# Quinones_Python_Portfilio-
This is the portfilio code that I learned during Bisc 450C


### This was my first Jupyter Notes Book Not sure why did not want to run on Praxis this was the one you helped me out in your office hours! I typed everything like on the videos.



```

    %matplotlib inline 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    sns.set(style = "darkgrid") 


```


    df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')


```


    df.head()


```


    df.tail()


```


    df.columns = ['year', 'rank', 'company', 'revenue', 'profit']



```


    df.head() 


```


    len(df)


```


    df.dtypes


```


    non_numeric_profits = df.profit.str.contains('[^0-9.-]')
    df.loc[non_numeric_profits].head()


```


    set(df.profit[non_numeric_profits]) 


```


    len(df.profit[non_numeric_profits])


```


    bin_size =plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006))


```


    df = df.loc[~non_numeric_profits]
    df.profit = df.profit.apply(pd.to_numeric)


```


    len(df)


```


    df.dtypes


```


    group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
    avgs = group_by_year.mean()
    x = avgs.index
    y1 = avgs.profit
    def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)


```


    fig, ax = plt.subplots()
    plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit             (millions)')


```


    y2 = avgs.revenue
    fig, ax = plt.subplots()
    plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue          (millions)')


```

    def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha=0.2)
    plot(x, y, ax, title, y_label)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'
    stds1 = group_by_year.std().profit.values
    stds2 = group_by_year.std().revenue.values
    plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
    plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
    fig.set_size_inches(14, 4)
    fig.tight_layout()


```


## This is python fundamentals my second NOTEBOOK


```

    # Any python interpreter can be used a calculator:
    3 + 5 * 4


```


    # Lets save a value to a variable 
    weight_kg = 60



```


    print(weight_kg)


```


    # weight = valid
    # 0weight = invalid
    # weight and Weight are different 


```


    #Types of data 
    #There are three common types of data 
    #Integer numbers 
    #floating point numbers 
    # Straings


```


    # Floating point number
    weight_kg = 60.3


```


    # String comprised of Letters
    patient_name = "Jon Smith"


```


    # String comprised of numbers
    patient_id = '001'


```


    # Use variables in python

    weight_lb = 2.2 * weight_kg

    print(weight_lb)


```


    # Lets add a prefix to our patient id 
    
    patient_id = 'inflam_' + patient_id
    
    print(patient_id)


```


    # Let combine print statements

    print(patient_id, 'weight in kilograms:', weight_kg)


```


    # we can call a function inside another function 
    
    print(type(60.3))
    
    print(type(patient_id))


```


    # We can also do calculations inside the print function 

    print('weight in lbs:', 2.2* weight_kg)


``` 


    print(weight_kg)


``


    weight_kg = 65.0
    print('weight in kilograms is now:', weight_kg)


## THIS NOTE BOOK IS ANALYZING DATA NOTEBOOK 


```


    import numpy


```


    numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')


```


    data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')


```


    print(type(data))


```


    print(data.shape)


```


    print('flirt value in data:', data[0,0])


```


    print('middle value in data:', data[29,19])


```


    print(data[0:4, 0:10])


```


    print(data[5:10, 0:10])


```


    small = data[:3, 36:]


```


    print(small)


```


    # let us a numpy function
    print(numpy.mean(data))


```


    maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)


```



    print(maxval)
    print(minval)
    print(stdval)


```


    maxval = numpy.amax(data)
    minval = numpy.amin(data)
    stdval = numpy.std(data)



```


    print(maxval)
    print(minval)
    print(stdval)


```


    print('maximum inflammation:', maxval)
    print('minimum inflammation:', minval)
    print('standard deviation:', stdval)



```



    #Somestimes we want to look at vairation in statistical values, such as maximum inflammation per       patient, 
    #or average from day one:

    patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)

    print('maximum inflammation for patient 0'), numpy.amax(patient_0)



```


    print('maximum inflammation for patient 2:', numpy.amax(data [2, :]))



```


    print(numpy.mean(data, axis = 0))
    

```



    print(numpy.mean(data, axis = 0).shape)



```


    print(numpy.mean(data, axis = 1))


### This is the third part of Analyzing Patient dat 



```


    import numpy
    data = numpy.loadtxt(fname= 'inflammation-01.csv', delimiter = ',')


```


    import matplotlib.pyplot
    image = matplotlib.pyplot.imshow(data)
    matplotlib.pyplot.show()


```


    # Average inflammation over time 

    ave_inflammation = numpy.mean(data, axis =0)
    ave_plot = matplotlib.pyplot.plot(ave_inflammation)
    matplotlib.pyplot.show()


```


    max_plot = matplotlib.pyplot.plot(numpy.amax(data,axis =0))
    matplotlib.pyplot.show()


```


    min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis = 0))
    matplotlib.pyplot.show()


```


    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))

    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)

    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))

    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))

    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))

    fig.tight_layout()

    matplotlib.pyplot.savefig('inflammation.png')
    matplotlib.pyplot.show()


## THIS IS THE STORING LIST NOTEBOOK 


```


    odds = [1, 3, 5, 7]
    print ('odds are:', odds)


```


    print('first element:', odds[0])
    print('last element:', odds[3])
    print('"-1" element:', odds[-1])


```


    names = ['Curie', 'Darwing', 'Turing'] # Typo in Darwin's name 

    print('name is originally:', names)

    names[1] = 'Darwin' # Correct the name 
  
    print('final values of names:', names)


```


    #name = 'Darwin'
    #name[0] = 'd'


```


    odds.append(11)
    print('odds after adding a value:', odds


```


    removed_element =odds.pop(0)
    print('odds after removing the first element:', odds)
    print('removed_element:', removed_element)


```


    odds.reverse()
    print('odds after reversing:', odds)


```


    odds = [3, 5, 7]
    primes = odds
    primes.append(2)
    print('primes:', primes)
    print('odds:', odds)


```



    odds = [3, 5, 7]
    primes = list(odds)
    primes.append(2)
    print('primes:', primes)
    print('odds:', odds)



```


    binomial_name = "Dropsophila melanogaster"
    group = binomial_name[0:10]
    print('group:', group)

    species = binomial_name[11:23]
    print('species:', species)

    chromosomes = ['X', 'Y', '2', '3', '4',]
    autosomes = chromosomes[2:5]
    print('autosomes:', autosomes)

    last = chromosomes[-1]
    print('last:', last)


```


    binomial_name = "Drosophila melanogaster"
    group = binomial_name[0:10]
    print('group:', group)

    species = binomial_name[11:23]
    print('species:', species)

    chromosomes = ['X', 'Y', '2', '3', '4',]
    autosomes = chromosomes[2:5]
    print('autosomes:', autosomes)

    last = chromosomes[-1]
    print('last:', last)


```


    date = 'Thursday 7 Novemeber 2024'
    day = date[0:8]
    print('Using 0 to being range:', day)
    day = date[:8]  
    print('Omitting beginning index', day)



```


    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    sond = months[8:12]
    print('With known last position:', sond)

    sond= months [8:len(months)] 
    print('Using len() to get last entry:', sond)

    sond = months[8:] 
    print('Omitting ending index:', sond)


```


## THIS IS THE LOOPS NOTEBOOK

```

    odds = [1, 3, 5, 7]


```


    print(odds[0])
    print(odds[1])
    print(odds[2])
    print(odds[3])


```


    odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    for nun in odds:
    print(nun)

```


    length = 0 
    names = ['Curie', 'Darwin', 'Turning']
    for value in names:
    length = length + 1
    print('There are', length, 'names in the list.')


```


    name = "Rosalind"
    for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
    print('after the loop, name is', name)


```


    print(len([0,1,3,4]))


```

    name = ['Curie', 'Darwin', 'Turing' ]

    print(len(name))

```


## THIS IS THE USING MULTIPLE FILES NOTBOOK


```


    import glob


```


    print(glob.glob('inflammation*.csv'))


```


    import glob 
    import numpy
    import matplotlib.pyplot
  
    filenames = sorted(glob.glob('inflammation*.csv'))
    filenames = filenames[0:3]

    for filename in filenames:
    print(filenames)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()


```


## 


```


    num = 37
    if num > 100:
    print('greater')
    else:
    print('not greater')
    print('done')


```


    num = 53
    print('before conditional...')
    if num > 100:
    print(num, 'is greater than 100')
    print('...after conditional')


```


    num = 14

    if num > 0:
    print(num, 'is positive')
    elif num == 0:
    print(num, 'is zero')
    else:
    print(num, 'is negative')


```


    if (1 > 0) and (-1 >= 0):
    print('both parts are true ')
    else:
    print('at least one part if false')


```


    if (1 > 0) or (-1 >= 0):
    print('at least one part if false')
    else:
    print('both of these are false')


```


    import numpy 


```


    data = numpy.loadtxt(fname='inflammation-02.csv', delimiter= ',')


```


    max_inflammation_0 = numpy.amax(data, axis=0)[0]


```


    max_inflammation_20 = numpy.amax(data, axis = 0)[20]

    if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima!')


```


    max_inflammation_20 = numpy.amax(data, axis = 0)[20]

    if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima!')



    elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero!')
    
    else:
    print('Seems OK!')


```


    data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter=',')

    max_inflammation_0 = numpy.amax(data, axis =0)[0]
    
    max_inflammation_20 = numpy.amax(data, axis =0)[20]

    if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
    elif numpy.sum(numpy.amin(data, axis=0)) ==0:
    print('Minima add to zero! -> HEALTHY PARTICIPANT ALERT!')
    else:
    print('Seems ok!')


```


## THIS IS THE CREATING FUNCTIONS FOLDER


```


    fahrenheit_val = 99 
    celsius_val = ((fahrenheit_val - 32) *(5/9))

    print(celsius_val)


```


    fahrenheit_val2 = 43
    celsius_val2 = ((fahrenheit_val2 -32) * (5/9))

    print(celsius_val2)



```


    def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp -32) * (5/9))
    # Return the values of the new variable 
    return converted


```


    def fahr_to_celsius(temp):
    # Return converted values more effeciently using the return function without creating 
    # a new variable. This code does the same thing as the previous function but it is more 
    # explitcit in explaining how the return command works.
    return ((temp - 32) *(5/9))


```


    explicit_fahr_to_celsius(32)


```


    explicit_fahr_to_celsius(32)


```


    print('Freezing point of water:', fahr_to_celsius(32), 'C')
    print('Boiling point of water:', fahr_to_celsius(212), 'C')


```


    def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

    print('freezing point of water in Kelvin:', celsius_to_kelvin (0.)) 


```

    def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k
    print ('boiling point of water in Kelvin:', fahr_to_kelvin (212.0))


```


    temp_kelvin = fahr_to_kelvin(212.0)
    print('Temperature in Kelvin was:', temp_kelvin)


```


      temp_kelvin


```


    def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
    temp_fahr = 212.0
    temp_kelvin = fahr_to_kelvin(temp_fahr)
    
    print_temperatures()



```


    import numpy
    import matplotlib
    import matplotlib.pyplot
    import glob


```


    'freezing point of water in Kelvin'
    def visualize(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()


```


    def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis=0)[20] ==20:
        print("Suspicious looking maxima!")
    elif numpy.sum(numpy.amin(data, axis=0)) == 0:
        print('Minia add up to zero!')
    else:
        print('Seems ok!')


```


    filenames = sorted(glob.glob('inflammation*.csv'))

    for filename in filenames[:3]:
    print(filename)
    visualize(filename)
    detect_problems(filename)


```


    def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value


```


    z = numpy.zeros((2,2))
    print(offset_mean(z,3))


```


    data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
  
    print(offset_mean(data, 0))



```


    print('original min, mean and max are', numpy.amin(data), numpy.mean(data), numpy.amax(data))
    offset_data = offset_mean(data, 0)
    print('min, mean, and max of offset data are:',
     numpy.amin(offset_data),
     numpy.mean(offset_data),
     numpy.amax(offset_data),)



```


    print('std dev before and after:', numpy.std(data), numpy.std(offset_data))


```


    print('difference in standard deviation before and after:',
     numpy.std(data) - numpy.std(offset_data))


```

    # offset_mean(data, target_mean_value):
    # return a new array containing the original data with its mean offset to match the desired value.

    def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value


```


    ef offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean to offset to match the desired        values"""
    return(data - numpy.mean(data)) + targted_mean_value


```


    help(offset_mean)


```


    def offset_mean(data, target_mean_value):
    """Return a new array containing the original data 
    with its mean offset to match the desired value.
    
    Examples
    ---------- 
    
    >>> Offset_mean([1,2,3],0)
    array([-1., 0., 1,]
    """
    return(data - numpy.mean(data)) + target_mean_value


```


    help(offset_mean)


```


    numpy.loadtxt('inflammation-01.csv', delimiter = ',')


```


    def offset_mean(data, target_mean_value):
    """Return a new array containing the original data 
    with its mean offset to match the desired value.
    
    Examples
    ---------- 
    
    >>> Offset_mean([1,2,3],0)
    array([-1., 0., 1,]
    """
    return(data - numpy.mean(data)) + target_mean_value


```


    test_data = numpy.zeros((2,2))
    print(offset_mean(test_data, 3)


```


    print(offset_mean(test_data, 0))


```


    def display(a=1, b=2, c=3):
    print('a:', a, 'b:', b, 'c:', c)
    
    print('no parameters:')
    display()
    print('one parameter:')
    display (55)
    print('two parameters:')
    display(55, 66)


```


    print('only setting the value of c')
    display(c = 77)



````


    help(numpy.loadtxt)


```


    numpy.loadtxt('inflammation-01.csv', delimiter =  ',')


```


    def s(p):
    a = 0 
    for v = in p:
        a+= v 
    m = a / len(p)
    d = 0
    for v in p:
        d+= (v - m) * (v - m)
    return numpy.sqrt(d / len(p) - 1))
    
    def std_dev(sample):
    sample_sum = 0
    for value in sample 
        sample_sum+= value
        
    sample_mean = sample_sum / len(sample)
    
    sum_squared_devs = 0
    for value in sample:
        sum_squared_dev += (value - sample_mean) * (value - sample_mean)
        
    return numpy.sqrt(sum_squared_devs / (len(sample) - 1))



## THIS IS THE DEFENSE PROGRAM AND DEBUGGING/ NONE BECAUSE ALL THE CODE WERE INCORRECT ON THE VIDEOS


### TRANSCRIBING DNA INTO RNA WITH PYTHON MY GENE WAS ALPHA_FETO_PROTEIN


```


    # Prompt the user to enter the fasta file name 

    input_file_name = input('Enter the name of the input fasta file: ')



```


    with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()


```


    rna_sequence = ""
    for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide


```


    #Prompt the user to enter the output file name 

    output_file_name = input('Enter the name of the output file: ')


```


    #Save the user to enter the output file name 

    with open(output_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print(f"The RNA sequence has been saved to {output_file_name}")


```


    print(rna_sequence)


## THIS TRANSLATION OF RNA INTO A PROTEIN OF ALPHA_FETO_PROTEIN


```

    # Promtpt the user to enter the input RNA filename 

    input_file_name = input('Enter the name of the input RNA file: ')


```


    # Open the input RNA file and read the RNA sequence

    with open(input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()


```


    # Define the codon table 

codon_table = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*", 
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R", 
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


```


    # Translate RNA to protein

    protein_sequence = " "
    for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid


```


    # Prompt the user to enter the output file name 

    output_file_name = input("Enter the name of the output file: ")


```


    with open(output_file_name, "w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name}")

```


    print(protein_sequence)


```

## THIS IS THE END OF MY PYTHON CODING THANK YOU VERY MUCH 
