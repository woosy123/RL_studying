# from neo4j import GraphDatabase

from wsgiref import simple_server
from concurrent import futures
import time
import logging
import os
import threading
import pandas as pd
import grpc
import message_pb2
import message_pb2_grpc
import random
# from metrics.collector import collector

import actions

NUM_WORKERS = 32



class InteractionServicer(message_pb2_grpc.InteractionServicer):
    """Provides methods that implement functionality of route guide server."""
    def __init__(self):
        self.collector  = self.stat_collector()
        self.tracing = self.get_stat_of()
        # self.read_tracing_stat = self.tracing_stat_reader()
    # request: ComponentId
    # response: ToClientMessage
    def GetState(self, request, context):
        metrics_stat = self.collector
        tracing_stat = self.tracing
        message = message_pb2.ToClientMessage()
        message.name = 'aa'
        message.node = 'bb'
        message.id = 'cc'
        message.usage.cpu = int(metrics_stat['cpu'])
        message.usage.memory = int(metrics_stat['memory'])
        message.usage.llc = 0 #metrics_stat['cache']
        message.usage.network = int(metrics_stat['network'])
        message.usage.io = 0 # metrics_stat['diskio']
        message.limit.cpu = 1
        message.limit.memory = 1
        message.limit.llc = 0
        message.limit.io = 0
        message.limit.network = 1
        message.other.slo_retainment = tracing_stat['slo_retainment']
        message.other.curr_arrival_rate = tracing_stat['curr_arrival_rate']
        message.other.rate_ratio = tracing_stat['rate_ratio'];
        for i in range(0,3):
            message.other.percentages.append(1.0) # tracing_stat['percentages'];
        message.status = 'OK'
        return message
    
    # request: ToServerMessage
    # response: ToClientMessage
    def PerformAction(self,request, context):
        # if request.id in self.container_map:
            # execute action
            actions.cpu(request.id, request.action.cpu) 
            actions.memory(request.id, request.action.memory) # , self.container_map[request.id].cores
            actions.network(request.id, request.action.network)
            # response
            metrics_stat = self.collector
            tracing_stat = self.tracing
            message = message_pb2.ToClientMessage()
            message.name = 'aa'
            message.id = 'cc'
            message.usage.cpu = int(metrics_stat['cpu'])
            message.usage.memory = int(metrics_stat['memory'])
            message.usage.llc = 0 # metrics_stat['cache']
            message.usage.network = int(metrics_stat['network'])
            message.usage.io = 0 # metrics_stat['diskio']
            message.limit.cpu = 1
            message.limit.memory = 1
            message.limit.llc = 0
            message.limit.io = 0
            message.limit.network = 1
            message.other.slo_retainment = tracing_stat['slo_retainment']
            message.other.curr_arrival_rate = tracing_stat['curr_arrival_rate']
            message.other.rate_ratio = tracing_stat['rate_ratio'] 
            for i in range(0,3):
                message.other.percentages.append(1.0) # tracing_stat['percentages'];
            message.status = 'OK'
            return message

    def stat_collector(self):
        stat={}
        df = pd.read_csv("metrics.txt", sep=' ',header=None, names=['val'])
        df = df.apply(pd.to_numeric, errors='ignore')
        stat['cpu'] = float(df['val'][0][:5])
        stat['memory'] = float(df['val'][1])
        stat['network'] = float(df['val'][2])
        return stat


    def get_stat_of(self):
        stat = {}
        result = pd.read_csv("usage.txt",sep=' ',header=None, names=['val'])
        stat['slo_retainment'] = float(result['val'][0])
        stat['curr_arrival_rate'] = int(result['val'][1])
        stat['rate_ratio'] = float(result['val'][2])
        return stat

    def close(self):
        self.driver.close()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=NUM_WORKERS))
    message_pb2_grpc.add_InteractionServicer_to_server(
        InteractionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
