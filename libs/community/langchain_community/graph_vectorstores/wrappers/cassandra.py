from langchain_community.graph_vectorstores.interfaces import VectorStoreForGraphInterface
from langchain_community.vectorstores.cassandra import Cassandra

class CassandraVectorStoreForGraph(Cassandra, VectorStoreForGraphInterface):
    pass
