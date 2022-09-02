import nonogram

from numpy import random

def printSol(sol, constraints):
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    print(nonogram.Game(nLines,  nColumns, sol.points))

def readRulesFile(fileName):
    with open(fileName) as rulesFile:
        readingLines = True
        lines   = []
        columns = []

        for fileLine in rulesFile:
            if(fileLine == '-\n'):
                readingLines = False
                continue

            rulesInFileLine = [[int(rule) for rule in fileLine.split()]]
            if(readingLines):
                lines   += rulesInFileLine
            else:
                columns += rulesInFileLine

    return nonogram.Rules(lines=lines, columns=columns)

def createConstraints(rules, nPopulation):
    nLines   = len(rules.lines)
    nColumns = len(rules.columns)
    nPoints  = 0

    # Count total number of colored points
    for line in rules.lines:
        for rule in line:
            nPoints += rule

    return (rules, nLines, nColumns, nPoints, nPopulation)

def fitness(sol, constraints):
    rules, nLines, nColumns, nPoints, nPopulation = constraints
    
    # Count how many rules it is following
    count = 0
    game  = nonogram.Game(nLines, nColumns, sol)
    board = sol

    # Count in lines in ascending order
    for lineIndex in range(nLines):
        rulesQtt = len(rules.lines[lineIndex])

        columnIndex = 0
        ruleIndex   = 0

        while columnIndex < nColumns or ruleIndex < rulesQtt:
            countSegment = 0
            currRule = rules.lines[lineIndex][ruleIndex] if ruleIndex < rulesQtt else 0

            while columnIndex < nColumns and not board[lineIndex*nColumns + columnIndex]:   #uncolored homes
                columnIndex += 1

            while columnIndex < nColumns and board[lineIndex*nColumns + columnIndex]:       #colored homes
                countSegment += 1
                columnIndex += 1

            t = pow(2, abs(countSegment - currRule))
            count -= abs(t)
            # count = -(pow(2 , -count))
            ruleIndex += 1

    # Count in columns in ascending order
    for columnIndex in range(nColumns):
        rulesQtt = len(rules.columns[columnIndex])

        lineIndex = 0
        ruleIndex = 0

        while lineIndex < nLines or ruleIndex < rulesQtt:
            countSegment = 0
            currRule     = rules.columns[columnIndex][ruleIndex] if ruleIndex < rulesQtt else 0

            while lineIndex < nLines and not board[lineIndex*nColumns + columnIndex]:
                lineIndex += 1

            while lineIndex < nLines and board[lineIndex*nColumns + columnIndex]:
                countSegment += 1
                lineIndex    += 1

            t = pow(2, abs(countSegment - currRule))
            count -= abs(t)

            # count = -(pow(2, -count))
            ruleIndex += 1
    # count *=2
    # count = -(pow(2, -count))
    # print(count)
    return count       #count difference between population states and rule numbers for each row and col


#For Second Fitness Function
def best_fitness(sol, combinations_row, combinations_column, constraints):

    rules, nLines, nColumns, nPoints, nPopulation = constraints

    # Count how many rules it is following
    count = 0
    game = nonogram.Game(nLines, nColumns, sol)
    board = sol

    for lineIndex in range(nLines):
        count_line = -99999999
        for comb in combinations_row[lineIndex]:

            temp = 0
            columnIndex = 0

            while columnIndex < nColumns:

                if(board[lineIndex*nColumns + columnIndex] != comb[columnIndex]):
                    temp -= 1

                columnIndex += 1

            if(count_line < temp):
                count_line = temp
                temp_comb = comb

        count += count_line

    for columnIndex in range(nColumns):
        count_column = -99999999
        for comb in combinations_column[columnIndex]:

            temp = 0
            lineIndex = 0

            while lineIndex < nLines:

                if (board[lineIndex * nColumns + lineIndex] != comb[lineIndex]):
                    temp -= 1

                lineIndex += 1

            if (count_column < temp):
                count_column = temp
                temp_comb = comb

        count += count_column

    return count