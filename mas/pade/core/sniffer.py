"""Framework for Intelligent Agents Development - PADE

The MIT License (MIT)

Copyright (c) 2019 Lucas S Melo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.misc.utility import display_message, start_loop

from pade.web import flask_server
from pade.web.flask_server import AgentModel, basedir


from twisted.internet.defer import inlineCallbacks
from twisted.internet import reactor
from twisted.enterprise import adbapi
from sqlalchemy import create_engine, MetaData, Table

from pickle import loads, dumps
import xml.etree.ElementTree as ET
import random
import os
import sys

# Configuração síncrona para metadados
ENGINE = create_engine('sqlite:///' + os.path.join(basedir, 'data.sqlite'))
METADATA = MetaData()
METADATA.bind = ENGINE
# SQLAlchemy 2.0: autoload=True foi removido, usar apenas autoload_with
MESSAGES = Table('messages', METADATA, autoload_with=ENGINE)
AGENTS = Table('agents', METADATA, autoload_with=ENGINE)

# Pool de conexões assíncrono usando twisted.enterprise.adbapi
DB_POOL = adbapi.ConnectionPool('sqlite3', os.path.join(basedir, 'data.sqlite'), check_same_thread=False)

class Sniffer(Agent):
    """This is the class that implements the AMS agent."""

    def __init__(self, host='localhost', port=8001, debug=False):
        self.sniffer_aid = AID('sniffer@' + str(host) + ':' + str(port))
        super(Sniffer, self).__init__(self.sniffer_aid, debug=debug)
        self.sniffer = {'name':str(host),'port':str(port)}      
        self.host = host
        self.port = port
        self.messages_buffer = dict()
        self.buffer_control = True
        self.agent_db_id = dict()

    def handle_store_messages(self):
        for sender, messages in self.messages_buffer.items():
            if self.agent_db_id.get(sender) == None:
                with ENGINE.connect() as conn:
                    r = conn.execute(AGENTS.select().where(AGENTS.c.name == sender))
                    a = r.fetchall()
                    if a != []:
                        agent_id = a[0][0]
                        self.agent_db_id[sender] = agent_id
                    else:
                        print('Agent does not exist in database: {}'.format(sender))
            else:
                pass

            for message in messages:
                receivers = ';'.join([i.localname for i in message.receivers])
                content = message.content
                if isinstance(content, ET.Element):
                    content = ET.tostring(content)

                # Prepara dados para inserção assíncrona
                message_data = {
                    'agent_id': self.agent_db_id[sender],
                    'sender': message.sender.name,
                    'date': message.datetime,
                    'performative': message.performative,
                    'protocol': message.protocol,
                    'content': content,
                    'conversation_id': message.conversation_id,
                    'message_id': message.messageID,
                    'ontology': message.ontology,
                    'language': message.language,
                    'receivers': receivers
                }

                reactor.callLater(random.uniform(0.1, 0.5), self.register_message_in_db, message_data)

                if self.debug:
                    display_message(self.aid.name, 'Message stored')

        self.messages_buffer = dict()
        self.buffer_control = True

    @inlineCallbacks
    def register_message_in_db(self, message_data):
        # Usando adbapi para operação assíncrona
        sql = """INSERT INTO messages 
                 (agent_id, sender, date, performative, protocol, content, 
                  conversation_id, message_id, ontology, language, receivers) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (
            message_data['agent_id'],
            message_data['sender'],
            message_data['date'],
            message_data['performative'],
            message_data['protocol'],
            message_data['content'],
            message_data['conversation_id'],
            message_data['message_id'],
            message_data['ontology'],
            message_data['language'],
            message_data['receivers']
        )
        yield DB_POOL.runOperation(sql, params)

    def react(self, message):
        super(Sniffer, self).react(message)
        if 'ams' not in message.sender.name:
            content = loads(message.content)
            if content['ref'] == 'MESSAGE':
                _message = content['message']
                if self.messages_buffer.get(message.sender.name) == None:
                    self.messages_buffer[message.sender.name] = [_message]
                else:
                    messages = self.messages_buffer[message.sender.name]
                    messages.append(_message) 

                if self.buffer_control:
                    reactor.callLater(5.0, self.handle_store_messages)
                    self.buffer_control = False

if __name__ == '__main__':
    sniffer = Sniffer(port=sys.argv[1])
    start_loop([sniffer])
