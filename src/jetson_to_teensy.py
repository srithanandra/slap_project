import serial
import serial.tools.list_ports
import struct
import time

# Constants
WRITE_FORMAT = '<12f12f?'  # 12 positions, 12 torques, 1 bool for estop
READ_FORMAT = '<21f'  # 3 for IMU, 3 for gyro, 3 for accel, 12 for motor positions
BAUDRATE = 100000

class JetsonTeensyBridge:
    def __init__(self):
        self.teensy_connection = None
        try:
            port = self.find_teensy_port()
            if not port:
                raise ConnectionError('Teensy not found')
            
            self.teensy_connection = serial.Serial(port, BAUDRATE, timeout=0.01)
            time.sleep(2)
            self.teensy_connection.reset_input_buffer()
            return
        except (serial.SerialException, ConnectionError) as e:
            print(f'Connection attempt failed: {e}')
            time.sleep(1)
        
        raise SystemExit('Could not establish stable connection.')

    def find_teensy_port(self):
        for port in serial.tools.list_ports.comports():
            if 'Teensy' in port.description or 'ttyACM' in port.device:
                return port.device
        return None

    def communicate(self, positions, torques, estop=False):
        if len(positions) != 12 or len(torques) != 12:
            raise ValueError('12 joint inputs required.')

        self.write(positions, torques, estop)
        result = self.read()
        return result

    def write(self, positions, torques, estop=False):
        if len(positions) != 12 or len(torques) != 12:
            raise ValueError('12 joint inputs required.')
            
        try:
            data = struct.pack(WRITE_FORMAT, *positions, *torques, estop)
            self.teensy_connection.write(data)
        except Exception as e:
            print(f'Write failure: {e}')

    def read(self):
        try:
            feedback_info = self.teensy_connection.read(84)
            if len(feedback_info) == 84:
                unpacked = struct.unpack(READ_FORMAT, feedback_info)
                result = {}
                result['roll'] = unpacked[0]
                result['pitch'] = unpacked[1]
                result['yaw'] = unpacked[2]
                result['gyro'] = unpacked[3:6]
                result['accel'] = unpacked[6:9]
                result['motor_positions'] = unpacked[9:21]
            return result

        except Exception as e:
            print(f'Read failure: {e}')
            return None

    def close(self):
        try:
            if self.teensy_connection:
                self.teensy_connection.close()
        except Exception:
            pass