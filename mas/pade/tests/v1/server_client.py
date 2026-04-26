"""
Teste de servidor/cliente Twisted - Atualizado para Python 3.8+ e Twisted 24.x+
"""

from twisted.internet import protocol, reactor

class Sniffer(protocol.Protocol):
    
    def __init__(self, factory):
        self.factory = factory
    
    def connectionMade(self):
        if self.factory.isClient:
            # Twisted 24.x+ requer bytes em vez de strings
            message = 'Connection is made with ' + str(self.transport.getPeer())
            self.transport.write(message.encode('utf-8'))
        else:
            # Twisted 24.x+ requer bytes em vez de strings
            self.transport.write(b'You are a client and you are connected!')
        
    def dataReceived(self, data):
        # Twisted 24.x+ - data já é bytes
        print(data.decode('utf-8') if isinstance(data, bytes) else data)
        self.transport.loseConnection()

class SnifferFactory(protocol.ClientFactory):
    
    def __init__(self, isClient):
        self.isClient = isClient
        
    def buildProtocol(self, addr):
        return Sniffer(self)

reactor.listenTCP(1235, SnifferFactory(isClient=False))
reactor.run()