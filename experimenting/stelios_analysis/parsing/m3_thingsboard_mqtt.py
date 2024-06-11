"""Thingsboard MQTT gateway instance for M3."""

from common_tb_mqtt import TbMqttClient

TB_HOST = '52.77.235.183'
TB_PORT = 1884
ACCESS_TOKEN = 'AoiIn9tSbBr25wQONRZE'

client = TbMqttClient()


def publish(serial, payload):
    client._publish_with_checks_and_retries(serial,0,payload)
    #client.publish_telemetry_with_device_connection(serial, payload)

    
def setup(pending_commands):
    client.setup_mqtt_client(pending_commands, access_token=ACCESS_TOKEN, tb_host=TB_HOST, tb_port=TB_PORT)
  

def start():
    client.start_mqtt_client()


def stop(serial):
    client.stop_mqtt_client(serial)


# def Token():
#    headers = {'Content-Type': 'application/json','Accept': 'application/json',}
#    data = '{"username":"tenant@thingsboard.org", "password":"tenant"}'
#    response = requests.post('http://10.0.1.64:8080/api/auth/login', headers=headers, data=data)
#    return response.json()['token']
