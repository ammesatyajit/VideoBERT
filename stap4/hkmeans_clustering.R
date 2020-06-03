args = commandArgs(trailingOnly=TRUE)

LEVELS <- strtoi(args[1])
CLUSTERS_PER_LEVEL <- strtoi(args[2])

print(sprintf("###### LEVELS = %d ######", LEVELS))
print(sprintf("###### CLUSTERS_PER_LEVEL = %d ######", CLUSTERS_PER_LEVEL))

ROOT_DIR <- sprintf("/home/david/clustering/results/clustering-L%d-C%d", LEVELS, CLUSTERS_PER_LEVEL)
print(sprintf("#### ROOT DIR: %s", ROOT_DIR))

ROOT_DATA_PATH <- file.path("/home/david/clustering/data/small.xdf")
print(sprintf("#### ROOT DATA PATH: %s", ROOT_DATA_PATH))

print("Creating root dir...")
dir.create(file.path(ROOT_DIR), showWarnings = FALSE)

VARS_TO_DROP <- c()
for(i in c(1:CLUSTERS_PER_LEVEL)) {
  VARS_TO_DROP <- c(VARS_TO_DROP, sprintf("rxCluster%d", i))
}

CLUSTER_SIZE_THRESHOLD <- CLUSTERS_PER_LEVEL

hcluster <- function(traceId, id, dataPath, numClusters, currentLevel, maxLevel) {
  LEVEL_ID <- sprintf("%s-%04d", traceId, id)
  LEVEL_ROOT_DIR <- file.path(ROOT_DIR, sprintf("L%d", currentLevel))
  dir.create(file.path(LEVEL_ROOT_DIR), showWarnings = FALSE)
  print(sprintf("#### ROOT LEVEL PATH: %s", LEVEL_ROOT_DIR))

  CLUSTER_ROOT_DIR <- file.path(LEVEL_ROOT_DIR, LEVEL_ID)
  print(sprintf("#### CLUSTER LEVEL PATH: %s", CLUSTER_ROOT_DIR))
  dir.create(file.path(CLUSTER_ROOT_DIR), showWarnings = FALSE)

  print("-------------------------------------------------------")
  print(sprintf("### STARTING hcluster L=%d, C=%d, ML=%d WITH DATA AT:", currentLevel, numClusters, maxLevel))
  print(dataPath)
  print("-------------------------------------------------------")
  print("")

  CLUSTER_INFO_SAVE_DIR <- file.path(sprintf("%s-%s", dataPath, "info"))
  dir.create(CLUSTER_INFO_SAVE_DIR, showWarnings = FALSE)
  
  # cluster this level
  clusters <- doCluster(dataPath, numClusters, id)

  saveClusterInfo(CLUSTER_INFO_SAVE_DIR, clusters)

  CLUSTER_FILES <- vector("list", numClusters)

  DROP_VARS <- c()
  names <- rxGetVarNames(data=dataPath)
  if("rxCluster0" %in% names) {
    # CAN ONLY OCCUR ON LEVEL 0
    DROP_VARS <- c("rxCluster0")
  } else if("rxCluster1" %in% names) {
    DROP_VARS <- VARS_TO_DROP
  }

  for(i in c(1:numClusters)) {
    clusterFilePath <- file.path(sprintf("%s", CLUSTER_ROOT_DIR), sprintf("cluster%d.xdf", i))
    print(cat("CLUSTER FILE PATH:", clusterFilePath)) 

    rxDataStep(inData=dataPath,
               outFile=clusterFilePath,
               reportProgress=1,
               xdfCompressionLevel=9,
               overwrite=TRUE,
               rowsPerRead=100000,
               rowSelection=parse(text=sprintf("rxCluster%d == %d", id, i)))
    CLUSTER_FILES[[i]] <- clusterFilePath
  }

  nextLevel <- currentLevel + 1
  if(nextLevel < maxLevel) {
    for(i in c(1:numClusters)) {
      path <- CLUSTER_FILES[[i]]
      summary <- rxSummary(formula=~V1, data=path)
      if(summary$nobs.valid > CLUSTER_SIZE_THRESHOLD) {
        hcluster(LEVEL_ID, i, path, numClusters, nextLevel, maxLevel)
      } else {
        print(sprintf("!!!!!!! ABORTING hcluster for cluster %d at level %d with size %d !!!!!!!", i, currentLevel, summary$nobs.valid))
      }
    }
  }
}

doCluster <- function(dataPath, numClusters, ColId) {
  dataXDF <- file.path(dataPath)

  # create formula
  names <- rxGetVarNames(data=dataXDF)
  names <- names[lapply(names, isVarName) == TRUE]
  print(sprintf("###### CLUSTERING WITH %d VARIABLES ######", length(names)))
  formula <- as.formula(paste0("~", paste(names, collapse="+")))

  clusters <- rxKmeans(formula=formula,
                       data=dataXDF,
                       numClusters=numClusters,
                       reportProgress=1,
                       blocksPerRead=1,
                       outFile=dataXDF,
                       outColName=sprintf('rxCluster%d', ColId),
                       maxIterations=10,
                       overwrite=TRUE
                      )

  clusters
}

saveClusterInfo <- function(saveDir, clusters) {
  write.table(clusters$centers, file=file.path(sprintf("%s", saveDir), "centers.csv"), col.names=FALSE, row.names=FALSE, sep=",") 
  data <- cbind(clusters$tot.withinss, clusters$betweenss, clusters$totss, clusters$valid.obs, clusters$missing.obs, clusters$betweenss/clusters$totss, Reduce("+", clusters$size))
  write.table(data, file=file.path(sprintf("%s", saveDir), "stats.csv"), col.names=FALSE, row.names=FALSE, sep=",")
  save(file=file.path(sprintf("%s", saveDir), "clusters.RData"), clusters)
}

isVarName <- function(val) {
  startsWith(val, "V")
}

hcluster("CLUSTERING", 0, ROOT_DATA_PATH, CLUSTERS_PER_LEVEL, 0, LEVELS)
