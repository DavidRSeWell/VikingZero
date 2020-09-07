import datetime
import pymongo

from dataclasses import dataclass

@dataclass
class ExperimentInfo:
    id: int
    env: str
    start_time: str
    end_time: str
    status: str
    loss_data: list
    games: dict
    config: dict
    def __str__(self):
        return f"ID={self.id} ENV={self.env} Stime={self.start_time} Etime={self.end_time} Status={self.status}"


class SacredDB:
    """
    Class for connecting to sacred experiments
    """
    def __init__(self,ip="localhost",port=27017,dbname=""):

        self._db = self.load_db(ip,port,dbname)
        self._dbname = dbname
        self._ip = ip
        self._port = port

    def load(self,id):
        runs = self._db.runs
        metrics = self._db.metrics
        exp = runs.find({"_id":id})[0]

        id = exp["_id"]

        env = exp["config"]["env"]

        s_time = exp["start_time"].strftime("%Y-%m-%d %H:%M:%S")

        if "stop_time" in exp.keys():
            e_time = exp["stop_time"].strftime("%Y-%m-%d %H:%M:%S")
        else:
            e_time = None
        status = exp["status"]
        try:
            loss_data = metrics.find({"run_id": id})[0]["values"]
        except:
            loss_data = []

        exp_info = exp["info"].copy()

        exp_config = exp["config"].copy()

        for key in exp["info"].keys():
            if "game_" not in key:
                del exp_info[key]

        Exp = ExperimentInfo(id, env, s_time, e_time, status, loss_data, exp_info,exp_config)
        return Exp

    def load_all(self):
        """
        Load all experiments
        :return:
        """
        assert self._db is not None

        runs = self._db.runs
        metrics = self._db.metrics

        exps = []
        for i in range(runs.count()):
            exp = runs.find()[i]
            id = exp["_id"]

            exp = self.load(id)
            exps.insert(0,exp)

        return exps

    def load_db(self,ip,port,dbname):

        client = pymongo.MongoClient(ip,port)
        return client[dbname]
