package com.company;

import java.io.IOException;
import java.util.Arrays;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;
import it.unimi.dsi.webgraph.NodeIterator;
import java.io.BufferedWriter;
import java.io.FileWriter;
public class Main {

    public static void main(String[] args) {
        String fileName="/home/ding/data/cnr-2000/cnr-2000.txt";
        final ProgressLogger pl = new ProgressLogger();
        final String basename = args[0];
        final ImmutableGraph graph;
            
        int curr, succ;
        System.out.println("Starting visit..." );

        try {
            graph = ImmutableGraph.load( basename, pl );

            final int n = graph.numNodes();
            final NodeIterator nodeIterator = graph.nodeIterator();
            
            BufferedWriter out=new BufferedWriter(new FileWriter(fileName));

            for( int i = 0; i < n; i++ ) {
                curr = nodeIterator.nextInt();

                LazyIntIterator successors;

                successors = graph.successors( curr );
                int d = graph.outdegree( curr );
                while(d-- != 0 ) {
                    succ = successors.nextInt();
                    out.write(curr + " " + succ);
                    out.newLine();
                }
            }           
                    
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
