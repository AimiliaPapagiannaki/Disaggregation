"""A module used to create a Thingsboard MQTT gateway instance.

See https://www.eclipse.org/paho/clients/python/docs/ for documentation."""

from datetime import datetime
import os
import json
from time import sleep

import paho.mqtt.client as mqtt


MAX_QUEUED_OUTGOING_MESSAGES = int(1e5)
MAX_INFLIGHT_MESSAGES = 30
MIN_RECONNECT_DELAY = 1
MAX_RECONNECT_DELAY = 10
KEEP_ALIVE_INTERVAL_SEC = 15

TOPIC_CONNECT = 'v1/gateway/connect'
TOPIC_DISCONNECT = 'v1/gateway/disconnect'
TOPIC_TELEMETRY = 'v1/gateway/telemetry'
TOPIC_ATTRIBUTE_UPDATES = 'v1/gateway/attributes'
TOPIC_RPC = 'v1/gateway/rpc'

LOG_FOLDER_NAME = 'logs'
LOG_LEVEL_STRING = {
    1: 'INFO',
    2: 'NOTICE',
    4: 'WARNING',
    8: 'ERROR',
    16: 'DEBUG'
}

# RC_CODE_DESCRIPTION = {
#     0: 'Connection successful',
#     1: 'Connection refused - incorrect protocol version',
#     2: 'Connection refused - invalid client identifier',
#     3: 'Connection refused - server unavailable',
#     4: 'Connection refused - bad username or password',
#     5: 'Connection refused - not authorised'
# }

HEX_DIGITS = frozenset('0123456789abcdefABCDEF')


class TbMqttClient(object):
    """Thingsboard MQTT gateway."""
    
    def __init__(self):
        self.mqtt_client = None  # it is initialized after client setup
        self.connected = False

    def setup_mqtt_client(self, pending_commands, access_token, tb_host, tb_port, client_id='', clean_session=False):
        if not clean_session and client_id == '':
            client_id = '{}_{}'.format("6b6742e0-7977-11ea-a32d-9198d6b065fe", access_token)
        mqtt_client = mqtt.Client(
            client_id=client_id,
            clean_session=clean_session,
            userdata=[pending_commands]
        )
        mqtt_client.max_inflight_messages_set(MAX_INFLIGHT_MESSAGES)
        mqtt_client.max_queued_messages_set(MAX_QUEUED_OUTGOING_MESSAGES)
        mqtt_client.reconnect_delay_set(min_delay=MIN_RECONNECT_DELAY, max_delay=MAX_RECONNECT_DELAY)
        mqtt_client.username_pw_set(username=access_token)

        mqtt_client.on_connect = self._on_connect
        mqtt_client.on_disconnect = self._on_disconnect
        mqtt_client.on_subscribe = self._on_subscribe

        mqtt_client.connect_async(host=tb_host, port=tb_port, keepalive=KEEP_ALIVE_INTERVAL_SEC)
        #mqtt_client.connect(host=tb_host, port=tb_port, keepalive=KEEP_ALIVE_INTERVAL_SEC)
        
        self.mqtt_client = mqtt_client
        
    def _friendly_log(self,log_string, log_level):
        print('{} [{}] {}'.format(datetime.utcnow(), log_level, log_string))
    
    def _publish_with_checks_and_retries(self, topic, qos, payload):
        # Retries are added due to infrequent but problematic errors caused by trying to publish when the client has
        # disconnected.
        while True:
            try:
                message_info = self.mqtt_client.publish(TOPIC_TELEMETRY, qos=qos, payload=payload)
                if message_info.rc != mqtt.MQTT_ERR_SUCCESS:
                    log_level = 'ERROR' if message_info.rc == mqtt.MQTT_ERR_QUEUE_SIZE else 'WARNING'
                    self._friendly_log(
                        log_string='Failed to publish message with rc {}: {}'.format(
                            message_info.rc,
                            mqtt.error_string(message_info.rc)
                        ),
                        log_level=log_level
                    )
                    self.mqtt_client.reconnect()
                break
            except Exception as exc:
                self._friendly_log(
                    log_string='Caught exception {} while publishing. Retrying...'.format(exc),
                    log_level='WARNING'
                )
                sleep(1)
        
    def _on_connect(self, client, userdata, flags, rc):
        """ The callback for when the client receives a CONNACK response from the server. """
        if rc == 0:
            self.connected = True
            self._friendly_log(
                log_string='Connected successfully',
                log_level='INFO'
            )
            # attempt to subscribe to relevant topics
            # client.subscribe([(TOPIC_ATTRIBUTE_UPDATES, 1), (TOPIC_RPC, 1)])
        elif 0 < rc <= 5:
            self.connected = False
            self._friendly_log(
                log_string='Failed to connect with result code {}: {}!'.format(rc, mqtt.connack_string(rc)),
                log_level='ERROR'
            )
        else:
            self.connected = False
            self._friendly_log(
                log_string='Unexpected response code {}: {} in CONNACK!'.format(rc, mqtt.connack_string(rc)),
                log_level='ERROR'
            )
        
    def _on_disconnect(self, client, userdata, rc):
        """ The callback for when the client disconnects from the server. """
        self.connected = False
        if rc == 0:
            self._friendly_log(
                log_string='Client has successfully disconnected',
                log_level='INFO'
            )
        else:
            self._friendly_log(
                log_string='Unexpected disconnection with rc {}: {}. Re-connecting...'.format(
                    rc,
                    mqtt.connack_string(rc)
                ),
                log_level='WARNING'
            )
            self.mqtt_client.reconnect()
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        if 128 in granted_qos:
            self._friendly_log(
                log_string='At least one subscription was denied with granted_qos = {}!'.format(granted_qos),
                log_level='ERROR'
            )
        else:
            self._friendly_log(
                log_string='Subscribed successfully',
                log_level='INFO'
            )
    
    def start_mqtt_client(self):
        #self.log_file = open(self.log_file_path, 'a', 1)  # line buffered
        self.log_file = open('Paho.txt', 'a', 1)
        self._friendly_log(log_string='### Client started network loop ###',log_level ='INFO')
        # start network loop
        self.mqtt_client.loop_start()
    
    def stop_mqtt_client(self , serial):
        if self.mqtt_client is not None:
            if self.connected:
                # disconnect devices
                self._publish_with_checks_and_retries(TOPIC_DISCONNECT, 1, json.dumps({'device': serial}))
                self.mqtt_client.disconnect()
                self.mqtt_client.loop_stop()
        self._friendly_log(
            log_string='### Client stopped ###',
            log_level='INFO'
        )
        sleep(1)  # waiting for the disconnection message to be logged before closing the file
        self.log_file.close()
