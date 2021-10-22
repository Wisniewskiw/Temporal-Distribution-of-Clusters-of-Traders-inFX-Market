library('SVN')
window_cluster_svn <- function(df,name,cut,windowsize,windowroll,bb,iter,statsforinfomap=1) {

 
column_names <- data.frame(month = "month", active_traders="active_traders",groups = "groups",links="#links",size="<size>",allingroups="allingroups")
write.table(column_names, file = paste(name,"min_stats_rollcut",cut,"cut.csv",sep=""), row.names = FALSE, 
            append = FALSE, col.names = FALSE, sep = ", ", quote = TRUE)

pb = txtProgressBar(min = 0, max =iter, initial = 0) 
for(i in 0:iter){
setTxtProgressBar(pb,i)  
  
a<- windowroll*i+bb
b<-windowsize+windowroll*i +bb
r_df<-df[a:b,]
l=(which_nas <- apply(r_df, 2, function(X) all(is.na(X))))
                      
is.nan.data.frame <- function(x)
do.call(cbind, lapply(x, is.na))

r_df[is.nan(r_df)] <- 2
 
r_df= as.matrix(r_df)
 
mylinks=SVN_links(r_df,states.pair.types = "same",exclude.states =c(2))


pathout="output\\dynamic\\"                      

if(statsforinfomap==1){
    myclusters=SVN_clusters(mylinks )

    newline <- data.frame(t(c(myclusters$modularity,length(l)-sum(l), length(myclusters),length(mylinks$i),mean(sizes(myclusters)),sum(sizes(myclusters)))))
    write.table(newline, file = paste(pathout,name,"min_stats_rollcut",cut,"cut.csv",sep=""), row.names = FALSE, 
              append = TRUE, col.names = FALSE, sep = ", ")
    
}                       

                      
 
#com<-data.frame(cluster = as.numeric(myclusters$membership) ,trader = myclusters$names) 

#savecommunity=paste(pathout,"community_",name,"min_stats_rollcut_window_",a,"_",b,"_",cut,"cut.csv",sep="")
#write.csv(com, file =savecommunity,sep="\t",row.names=F )  
                      
write.csv(data.frame(mylinks) , file =paste(pathout,"links_",name,"min_stats_rollcut_window_",a,"_",b,"_",cut,"cut.csv",sep=""),sep="\t",row.names=F )                       
                      
}
}

getSVNlinks<-function(str,cut,windowsize,windowroll,b,iter){
window_cluster_svn(r_from_pd_df10,paste(str,"10",sep=""),cut,windowsize*72,windowroll*72,b*72,iter)
window_cluster_svn(r_from_pd_df15,paste(str,"15",sep=""),cut,windowsize*48,windowroll*48,b*48,iter)
window_cluster_svn(r_from_pd_df30,paste(str,"30",sep=""),cut,windowsize*24,windowroll*24,b*24,iter)
window_cluster_svn(r_from_pd_df60,paste(str,"60",sep=""),cut,windowsize*12,windowroll*12,b*12,iter)
window_cluster_svn(r_from_pd_df120,paste(str,"120",sep=""),cut,windowsize*6,windowroll*6,b*6,iter)
window_cluster_svn(r_from_pd_df180,paste(str,"180",sep=""),cut,windowsize*4,windowroll*4,b*4,iter)
window_cluster_svn(r_from_pd_df360,paste(str,"360",sep=""),cut,windowsize*2,windowroll*2,b*2,iter)
window_cluster_svn(r_from_pd_df1440,paste(str,"1440",sep=""),cut,windowsize,windowroll,b,iter)

}


 
####################### get_graph_links function  getting names of files where the adjascency matric is stored ####################################### 
get_graph_links<- function(name,minutes,cut,window,mypath){
    l=list.files(mypath)
    x=Filter(function(x) (grepl(paste("_",minutes,"min",sep=""), x)==1)& (grepl(name, x)==1) & (grepl(window, x)==1)& (grepl(paste(cut,"cut",sep=""), x)==1)& (grepl("links", x)==1) , l) 
    x=x[order(as.numeric(gsub("\\D", "", x)))]
    return(x)  
}

#######################  load_graph_links function  loading dataframe  where the adjascency matric is stored ####################################### 
 
load_graph_links<- function(path,name){
    dataframe <- read.csv(paste(path,name,sep=""),  header= TRUE)
    return(dataframe)}   
#######################   clustering function ####################################### 

clusterise <- function(df,method) {
    mygraph = graph_from_data_frame(df, directed = FALSE)
    if (method == "infomap") {myclusters = cluster_infomap(mygraph, nb.trials = 10)}
    else if ( method == "louvain") {myclusters = cluster_louvain(mygraph)}
    else if ( method == "edge_betweenness") {myclusters = cluster_edge_betweenness(mygraph,directed = FALSE)}
    else if (method == "fast_greedy") { mygraph=simplify( mygraph)
        myclusters = cluster_fast_greedy(mygraph)}
    else if (method == "label_prop") {myclusters = cluster_label_prop(mygraph)}
    else if (method == "leading_eigen") {myclusters = cluster_leading_eigen(mygraph,options=list(maxiter=1000000))}
    else if (method == "optimal") {myclusters = cluster_optimal(mygraph)}
    else if (method == "spinglass") {
        V(mygraph)$label <- seq(vcount(mygraph))
        graphs <- decompose.graph(mygraph)
        largest <- which.max(sapply(graphs, vcount))      
        myclusters = cluster_spinglass(graphs[[largest]])}
    else if (method == "walktrap") {myclusters = cluster_walktrap(mygraph)}
    else {
        warning("SVN_cluster: unknown method ", method, 
            ", using infomap")
        myclusters = cluster_infomap(mygraph, nb.trials = 10)
    }   
  return(myclusters)
}   
 
#######################   main function that clusters and saves  output in dataframes like traders and cluster ####################################### 

clusterise_and_save<- function(name,minutes,cut,window,mypath,method,output_path="output/dynamic_clusters/community_"){
    liste=get_graph_links(name,minutes,cut,window,mypath)
    iter=length(liste)
    
    print(paste(name,"****min:",minutes,"****cut:",cut,"****window:",window,"****path:",mypath,"****method:",method,sep=" "))
    pb = txtProgressBar(min = 1, max =iter, initial = 1) 
    
    for(i in 1:iter){
        setTxtProgressBar(pb,i) 
        d<-load_graph_links(mypath,liste[i]) 
        
        if(nrow(d)==0){
            com<-data.frame(cluster = NULL ,trader = NULL)
            savecommunity=paste(output_path,method,"_",liste[i],sep="")
            write.csv(com, file =savecommunity,sep="\t",row.names=F )  
        }
        else{
            myclusters<-clusterise(d[c("i","j")],method)
            com<-data.frame(cluster = as.numeric(myclusters$membership) ,trader = as.character(myclusters$names))
            savecommunity=paste(output_path,method,"_",liste[i],sep="")
             
            write.csv(com, file =savecommunity,sep="\t",row.names=F )    
            
            }
    }   
}        

createRclusters<-function(name,minutes_liste,cut_liste,window_liste,mypath_liste,cluster_methods_liste){

 
for(mypath in mypath_liste){
    for(window in window_liste){         
        for( cut in cut_liste){
            for( mins in minutes_liste){
                for( method in cluster_methods_liste){
                    clusterise_and_save(name,mins,cut,window,mypath,method)
                }
            }
        }
    }
 }


}

