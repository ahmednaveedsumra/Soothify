import pyfirmata
import time

def func1():
    board = pyfirmata.Arduino('COM3')
    while True:
        board.digital[12].write(1)
        time.sleep(1)
        board.digital[12].write(0)
        time.sleep(1)


def func2():
    board = pyfirmata.Arduino('COM3')
    while True:
        board.digital[11].write(1)
        time.sleep(1)
        board.digital[11].write(0)
        time.sleep(1)


def func3():
    board = pyfirmata.Arduino('COM3')
    while True:
        board.digital[10].write(1)
        time.sleep(1)
        board.digital[10].write(0)
        time.sleep(1)


