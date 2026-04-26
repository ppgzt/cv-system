# -*- coding: utf-8 -*-
"""
Teste de cliente Twisted - Atualizado para Python 3.8+ e Twisted 24.x+
"""

from twisted.internet import protocol, reactor

class Echo(protocol.Protocol):
    def connectionMade(self):
        # Twisted 24.x+ requer bytes em vez de strings
        self.transport.write(b'Hello Server!')
        print('Connection established!')

    def dataReceived(self, data):
        # Python 3 e Twisted 24.x+ - data é bytes
        print(data.decode('utf-8') if isinstance(data, bytes) else data)

class EchoFactory(protocol.ClientFactory):
    def buildProtocol(self, addr):
        return Echo()

    def clientConnectionFailed(self, connector, reason):
        print('Connection failed!')
        print(reason.getErrorMessage())
        reactor.stop()

if __name__ == '__main__':
    print('Connecting...')
    reactor.connectTCP('192.168.0.32', 1234, EchoFactory())
    reactor.run()