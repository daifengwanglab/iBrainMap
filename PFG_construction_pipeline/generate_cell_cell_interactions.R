library(reticulate)
library(parallel)
library(CellChat)
library(Seurat)
library(Matrix)
library(reshape2)

sp <- import("scipy.sparse")

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)

gene_expr_file = args[1] #e.g.='./file_path/sample1.npz'
cell_metadata_file = args[2] #e.g.='./file_path/sample1_meta_obs.csv'
gene_list_file = args[3] #e.g.='./file_path/gene_list.txt'
cell_barcode_column = args[4] #e.g.='barcodekey'
cell_group_column = args[5] # e.g.='subclass'
output_path = args[6] # e.g.="output_file"

species='human'
if (length(args) == 7) {
  species = args[7]
}

getIndividualNw <- function(gene_expr_file, cell_metadata_file, gene_list_file, 
                            cell_barcode_column, cell_group_column, output_path, species=species) {
  # Get donor name to use it as file output
  donor = gsub('.npz', '', gene_expr_file)
  
  # Read the cell metatadata information.
  #Cell metadata file should contain atleast two columns: barcode column and cell_group column
  meta = read.csv(cell_metadata_file, header=T)
  meta = meta[, c(cell_barcode_column, cell_group_column)]
  colnames(meta) =  c('barcodekey', cell_group_column)
  meta$barcodekey = gsub('-', '.', meta$barcodekey)
  rownames(meta) = meta$barcodekey
  
  # The npz file contains cell by genes. We read it and transpose it to gene by cells
  expr = as.matrix(sp$load_npz(gene_expr_file))
  expr = t(expr)
  # Log normalization
  expr = log(expr + 1)
  # Adding names of genes and cells
  rownames(expr) = gene_list$V1
  colnames(expr) = meta$barcodekey
  #Creating a sparse matrix
  pcells = Matrix(expr, sparse=T)

  rm(expr)
  gc()

  # Run Cellchat
  cellchat = runCellChat(pcells, meta, cell_group_column, species=species, fname=paste0(output_path, '/', donor))
}


runCellChat <- function(rna_data, phen, cellGrp_name, species='human', fname='test') {
  #Create a CellChat object from a data matrix
  cellchat <- createCellChat(object = rna_data, meta = phen, group.by = cellGrp_name)
  
  # Set the ligand-receptor interaction database
  if (species == 'mouse') {
    CellChatDB <- CellChatDB.mouse
  } else {
    CellChatDB <- CellChatDB.human
  }
  
  # use all CellChatDB for cell-cell communication analysis
  CellChatDB.use <- CellChatDB # simply use the default CellChatDB
  
  # Set the database to use
  cellchat@DB <- CellChatDB.use
  
  # subset the expression data of signaling genes for saving computation cost
  cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database

  #future::plan("multiprocess", workers = 40) # do parallel
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  
  # Compute the communication probability and infer cellular communication network
  cellchat <- computeCommunProb(cellchat, nboot=1000)
  
  # Filter out the cell-cell communication if there are only few number of cells in certain cell groups
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  
  # Compute aggregated cell-cell communication network 
  cellchat = aggregateNet(cellchat)
  
  int_wt1 = data.frame(cellchat@net$count)
  int_wt2 = data.frame(cellchat@net$weight)

  df.net <- subsetCommunication(cellchat)
  
  write.csv(int_wt1, paste0(fname, '_ctype_comm_nw_count.csv'), row.names = F, quote=F)
  write.csv(int_wt2, paste0(fname, '_ctype_comm_nw_weight.csv'), row.names = F, quote=F)
  write.csv(df.net, paste0(fname, '_lr_comm_nw.csv'), row.names = F, quote=F)

  return(cellchat)
}


