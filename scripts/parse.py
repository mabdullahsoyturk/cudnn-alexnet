with open("output.txt") as out:
    lines = out.readlines()

    total = 0

    for idx, line in enumerate(lines):
        times = line.split(",")[:-1]

        sum_of_line = 0

        for time in times:
                sum_of_line += float(time)

        if idx > 10:
            total += sum_of_line

    print(total)
