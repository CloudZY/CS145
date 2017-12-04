# =======================================================================
import random
import math

# =======================================================================
class DataPoints:
    # -------------------------------------------------------------------
    def __init__(self, keywords, tweet_id):
        self.keywords = keywords
        self.tweet_id = tweet_id
        self.isAssignedToCluster = False
        
    # -------------------------------------------------------------------
    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.keywords) + "," + str(pt.tweet_id) + ",0" + "\n")
        for w in range(len(clusters)):
            print("Cluster " + str(w+1) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.keywords) + "," + str(point.tweet_id) + "," + str((w + 1)) + "\n")
        f.close()
# =======================================================================
class DBSCAN:
    # -------------------------------------------------------------------
    def __init__(self):
        self.e = 0.0
        self.minPts = 2
        self.noOfLabels = 0
    # -------------------------------------------------------------------
    def readDataSet(self,filePath):
        dataSet = []
        with open(filePath) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            all_data = line.split('\t')
            keywords = all_data[:-1]
            tweet_id = all_data[-1]
            point = DataPoints(keywords,tweet_id)
            dataSet.append(point)
        return dataSet
    # -------------------------------------------------------------------
    def main(self, args):
        seed = 71

        dataSet = self.readDataSet("predict_class_1.vectors")
        #random.Random(seed).shuffle(dataSet)
        self.e = 0
        print("Esp :" + str(self.e))
        self.dbscan(dataSet)
        
    # -------------------------------------------------------------------
    def dbscan(self, dataSet):
        clusters = []
        visited = set()
        noise = set()

        # Iterate over data points
        for i in range(len(dataSet)):
            point = dataSet[i]
            if point in visited:
                continue
            visited.add(point)
            N = []
            minPtsNeighbours = 0

            # check which point satisfies minPts condition 
            for j in range(len(dataSet)):
                if i==j:
                    continue
                pt = dataSet[j]
                dist = self.getCosDistance(point.keywords, pt.keywords)
                if (1-dist) <= self.e:
                    minPtsNeighbours += 1
                    N.append(pt)

            if minPtsNeighbours >= self.minPts:
                cluster = set()
                cluster.add(point)
                point.isAssignedToCluster = True

                j = 0
                while j < len(N):
                    point1 = N[j]
                    minPtsNeighbours1 = 0
                    N1 = []
                    if not point1 in visited:
                        visited.add(point1)
                        for l in range(len(dataSet)):
                            pt = dataSet[l]
                            dist = self.getCosDistance(point.keywords, pt.keywords)
                            if (1-dist) <= self.e:
                                minPtsNeighbours1 += 1
                                N1.append(pt)
                        if minPtsNeighbours1 >= self.minPts:
                            self.removeDuplicates(N, N1)
                        else:
                            N1 = []
                    # Add point1 is not yet member of any other cluster then add it to cluster
                    if not point1.isAssignedToCluster:
                        cluster.add(point1)
                        point1.isAssignedToCluster = True
                    j += 1
                # add cluster to the list of clusters
                clusters.append(cluster)

            else:
                noise.add(point)

            N = []

        # List clusters
        print("Number of clusters formed :" + str(len(clusters)))
        print("Noise points :" + str(len(noise)))

        DataPoints.writeToFile(noise, clusters, "cos_DBSCAN.csv")
    # -------------------------------------------------------------------
    def removeDuplicates(self, n, n1):
        for point in n1:
            isDup = False
            for point1 in n:
                if point1 == point:
                    isDup = True
            if not isDup:
                n.append(point)
    # -------------------------------------------------------------------
    def getCosDistance(self,k1, k2):
        xy_sum = 0
        xx_sum = 0
        yy_sum = 0
        for i in range(len(k1)):
            xx_sum += float(k1[i])*float(k1[i])
            yy_sum += float(k2[i])*float(k2[i])
            xy_sum += float(k1[i])*float(k2[i])
        return xy_sum/math.sqrt(xx_sum*yy_sum)
# =======================================================================
if __name__ == "__main__":
    d = DBSCAN()
    d.main(None)
