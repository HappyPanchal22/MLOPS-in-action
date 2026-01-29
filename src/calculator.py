
def addition(x, y):
    return x + y

def subtraction(x, y):
    return x - y

def multiplication(x, y):
    return x * y

def division(x, y):
    if y == 0:
        return "Error: Division by zero"
    return x/ y

def main():
    x = int(input("Enter value of x:"))
    y = int(input("Enter value of y:"))

    print("Add", addition(x,y))
    print("Subtract", subtraction(x,y))
    print("Product", multiplication(x,y))
    print("Divide", division(x,y))


if __name__ == "__main__":
    main()

