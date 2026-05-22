from jetson_to_teensy import JetsonTeensyBridge

def main():
    connection = JetsonTeensyBridge()
    pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        while True:
            connection.communicate(positions=pos, torques=tor, estop=False)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: closing connection')
        connection.close()

if __name__ == '__main__':
    main()