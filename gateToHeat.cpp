/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: gudeh
 *
 * Created on 14 de maio de 2022, 14:25
 */

#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>



using namespace std;

#include <iostream>
#include <unordered_set>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>


using point_type   = boost::geometry::model::d2::point_xy<double>;
using box_type     = boost::geometry::model::box<point_type>;
using rtree_node_type = std::pair<box_type, std::string>;
using rtree_type      = boost::geometry::index::rtree<rtree_node_type, boost::geometry::index::rstar<16> >;

void show_rtree()
{
    
cout <<  "  ____________________                    " << endl;
cout <<  "  |    | b   |        |                   " << endl;
cout <<  "  |    |_____|        |                   " << endl;
cout <<  "  |     _____      ___|__       _____     " << endl;
cout <<  "  |    |  a  |    |   | C |    |  d  |    " << endl;
cout <<  "  |    |_____|    |___|___|    |_____|    " << endl;
cout <<  "  |___________________|______             " << endl;
cout <<  "      |  e  |         |  f  |             " << endl;
cout <<  "      |_____|         |_____|             " << endl;
cout << endl;

  
    auto box_a = box_type{point_type{5,5}, point_type{6,6}}; //inside
    auto box_b = box_type{point_type{3,8}, point_type{5,10}}; //corner top
    auto box_c = box_type{point_type{9,5}, point_type{11,6}}; //cross right
    auto box_d = box_type{point_type{15,2}, point_type{17,3}}; //out
    auto box_e = box_type{point_type{4,-2}, point_type{6,0}}; //out
    auto box_f = box_type{point_type{10,-2}, point_type{12,0}}; //out

    auto box_search = box_type{point_type{0,0}, point_type{10,10}};

    rtree_type rtree;

    rtree.insert(rtree_node_type{box_a, "box_a"});
    rtree.insert(rtree_node_type{box_b, "box_b"});
    rtree.insert(rtree_node_type{box_c, "box_c"});
    rtree.insert(rtree_node_type{box_d, "box_d"});
    rtree.insert(rtree_node_type{box_e, "box_e"});
    rtree.insert(rtree_node_type{box_f, "box_f"});
    
    cout << "intersect_nodes: " << endl;
    std::vector<rtree_node_type> intersect_nodes;
    rtree.query(boost::geometry::index::intersects(box_search), std::back_inserter(intersect_nodes));
    for (auto node : intersect_nodes) {
        cout << node.second << endl;
    }
    cout << endl;

    cout << "covered_by_nodes: " << endl;
    std::vector<rtree_node_type> covered_by_nodes;
    rtree.query(boost::geometry::index::covered_by(box_search), std::back_inserter(covered_by_nodes));
    for (auto node : covered_by_nodes) {
        cout << node.second << endl;
    }
    cout << endl;

    cout << "intersects_nodes: " << endl;
    std::vector<rtree_node_type> intersects_nodes;
    rtree.query(boost::geometry::index::intersects(box_search), std::back_inserter(intersects_nodes));
    for (auto node : intersects_nodes) {
        cout << node.second << endl;
    }
    cout << endl;

    cout << "overlaps_nodes: " << endl;
    std::vector<rtree_node_type> overlaps_nodes;
    rtree.query(boost::geometry::index::overlaps(box_search), std::back_inserter(overlaps_nodes));
    for (auto node : overlaps_nodes) {
        cout << node.second << endl;
    }
    cout << endl;

    cout << "disjoint_nodes: " << endl;
    std::vector<rtree_node_type> disjoint_nodes;
    rtree.query(boost::geometry::index::disjoint(box_search), std::back_inserter(disjoint_nodes));
    for (auto node : disjoint_nodes) {
        cout << node.second << endl;
    }
    cout << endl;

    cout << "intersect_with_area: (intersects + overlaps)" << endl;
    std::vector<rtree_node_type> int_area_nodes;
    rtree.query(boost::geometry::index::intersects(box_search), std::back_inserter(int_area_nodes));
    rtree.query(boost::geometry::index::overlaps(box_search), std::back_inserter(int_area_nodes));
    for (auto node : int_area_nodes) {
        cout << node.second << endl;
    }
    cout << endl;
    
}

class gate{
//    string name="";
//    float x0=0.0;
//    float y0=0.0;
//    float x1=0.0;
//    float y1=0.0;
//    vector<int> gatePos;
//    float width=0.0;
//    float height=0.0;
public:
    gate(){
        this->heatPlacement = -1.0;
        this->heatRouting = -1.0;
        this->heatPower = -1.0;
        this->heatIRDrop = -1.0;
        this->id = -1;
        this->conCount = -1;
        this->type = "NULL";
    }
    double heatPlacement;
    double heatRouting;
    double heatIRDrop;
    double heatPower;
    int id;
    string type;
    int conCount;
    
    void printGate(){
        cout << "id:" << this->id << ", type:" << this->type << ", conCount:"<< this->conCount << "heats(plac,rout,ir,pow):" 
             << heatPlacement << "," << heatRouting<< ","  << heatIRDrop << "," << heatPower << endl;
    }
    
    void printGate( fstream &writer, string name ){
        writer << this->id << "," << name << "," << this->type << "," << this->conCount << "," 
             << heatPlacement << "," << heatRouting<< ","  << heatIRDrop << "," << heatPower << "\n" ;
    }
};

struct heatBbox{
    string name="";
    float x0=0.0;
    float y0=0.0;
    float x1=0.0;
    float y1=0.0;
    float width=0.0;
    float height=0.0;
    float value=0.0;
};

//Using netbeans to compile on proper folder. Original: ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/gatetoheat
int main(int argc, char** argv) {

    //TODO: p is analogous to "./myStuff/myDataSet/, no need for loop
    string root="./myStuff";
    vector<string> all_projects;
    vector<string> projectWithErrors;
    for( auto& p : filesystem::directory_iterator( root ) )
    {
        if ( p.is_directory() && p.path().string().find( "myDataSet" ) == string::npos)
            all_projects.push_back( p.path().string() );
    }
    
    for ( auto& project : all_projects )
    {
        cout<< "\n\nExecuting project:" << project << endl;
        string cellsPath = "";//, edgesPath;
        for ( auto &path : filesystem::directory_iterator( project ) )
        {
            if ( path.path().extension() == ".csv" && path.path().string().find("DGLcells") != string::npos )
                cellsPath = path.path().string();
        }
        
        vector<string> heat_csvs, position_files; 
        for ( auto &path : filesystem::directory_iterator( project ) )
        {
            if ( path.path().extension() == ".csv" && path.path().string().find("routingHeat") != string::npos )
                heat_csvs.push_back( path.path().string() );
            if ( path.path().extension() == ".csv" && path.path().string().find("placementHeat") != string::npos )
                heat_csvs.push_back( path.path().string() );
            if ( path.path().extension() == ".csv" && path.path().string().find("powerHeat") != string::npos )
                heat_csvs.push_back( path.path().string() );
            if ( path.path().extension() == ".csv" && path.path().string().find("irdropHeat") != string::npos )
                heat_csvs.push_back( path.path().string() );                
            
            if (path.path().extension() == ".csv" && path.path().string().find("gatesPosition") != string::npos)
            {
                cout << "gates file:" << path.path().stem().string() << endl;
                position_files.push_back( path.path().string() );
            }
        }
        
        if( position_files.size() != 1 )
        {
            cout<<"ERROR, position_files !=1 (gates position files)"<<endl;
            projectWithErrors.push_back( project );
            continue;
        }
        if( heat_csvs.size() < 1 )
        {
            cout<<"ERROR, heat_csvs <1 (heats files)"<<endl;
            projectWithErrors.push_back( project );
            continue;
        }
        if( cellsPath == "" )
        {
            cout<<"ERROR, cellsPath empty!"<<endl;
            projectWithErrors.push_back( project );
            continue;
        }
        
        
//        std::map< std::string, std::array< double, 4 > > gate_to_heat;
        map< string, gate > gate_to_heat;
        
        //Reading file from logic synthesis (yosys)
        fstream dglCellsFile( cellsPath, ios::in );
        string myLine;
        getline( dglCellsFile, myLine );
        while  ( getline( dglCellsFile, myLine ) )
        {
            string word;
            stringstream s( myLine );
            vector<string> row;
            while ( getline( s, word, ',' ))
                row.push_back( word );
            gate myGate;
            myGate.id = stod( row[0] );
            //ignore name and hashid: [1] and [2]
            myGate.type = row[3];
            myGate.conCount = stod( row[4] );
            gate_to_heat.insert( pair< string, gate > ( row[1], myGate ) );
        } 
        
        rtree_type rtree;
        cout<<"position_files[0]: "<<position_files[0]<<endl;
        fstream filePositions( position_files[0], ios::in );
        getline( filePositions, myLine );
        while  ( getline( filePositions, myLine ) )
        {
            string word;
            stringstream s( myLine );
            vector<string> row;
            while ( getline( s, word, ',' ))
                row.push_back( word );
//            cout<<"row:";
//            for ( auto & R : row )
//                cout<<R<<",";
//            cout<<endl;
            
            if( gate_to_heat.find(row[0]) != gate_to_heat.end() )
            {
                auto box_a = box_type{ point_type{ stod(row[1]), stod(row[2]) }, point_type{ stod(row[3]), stod(row[4]) } };            
                rtree.insert( rtree_node_type{ box_a, row[0] } );
            }
//          TODO: what  todo when gates added by openroad/yosys? ignoring for now.            
//            else
//                cout<< "Gate added by OpenRoad: " << row[0] << ",pos:" << row[1] << "," << row[2] << "|" << row[3] << "," << row[4] << endl;
                
//            gate_to_heat.insert( pair< std::string, std::array< double, 4 > > ( row[0], std::array< double, 4 > { -1.0, -1.0, -1.0, -1.0 } ) );
        }
        

        for ( auto& heatFileName : heat_csvs )
        {
            string myLine, word;
            fstream fHeat( heatFileName, ios::in );
            getline( fHeat, myLine );
            while( getline( fHeat, myLine ) )
            {
                stringstream s( myLine );
                vector<string> row;
                while ( getline( s, word, ',' ) )
                    row.push_back( word );
//                cout<<"row:";
//                for ( auto & R : row )
//                    cout<<R<<",";
//                cout<<endl;
                
                auto box_search = box_type{ point_type{ stod(row[0]), stod(row[1])}, point_type{ stod(row[2]), stod(row[3]) } };
                std::vector<rtree_node_type> int_area_nodes;
                rtree.query( boost::geometry::index::intersects( box_search ), std::back_inserter( int_area_nodes ));
                for ( auto& intersect : int_area_nodes )
                {
//                    gate_to_heat.insert( std::pair< std::string, double >( intersect.second, stod( row[4] ) ) );
                    if ( heatFileName.find("routingHeat") != string::npos )
//                        gate_to_heat[ intersect.second ].at(0) = stod( row[4] );
                        gate_to_heat[ intersect.second ].heatRouting = stod( row[4] );
                    if ( heatFileName.find("placementHeat") != string::npos )
                        gate_to_heat[ intersect.second ].heatPlacement = stod( row[4] );
                    if ( heatFileName.find("powerHeat") != string::npos )
                        gate_to_heat[ intersect.second ].heatPower = stod( row[4] );
                    if ( heatFileName.find("irdropHeat") != string::npos )
                        gate_to_heat[ intersect.second ].heatIRDrop = stod( row[4] );
                }
            }
        }
//        std::cout << "heatFileName:" << heatFileName <<endl;
//        std::string outName = heatFileName.erase( 0, heatFileName.find_last_of("/") );
//        outName = outName.erase( outName.find_last_of("."), outName.size() );
        std::string outName = project+"/gatesToHeat.csv";
        std::cout << "outName:" << outName <<endl;
        fstream myOut( outName, ios::out );
//            std::map< std::string, double >::iterator it;
        myOut<<"id,name,type,conCount,placementHeat,routingHeat,irDropHeat,powerHeat\n";
        for( auto it = gate_to_heat.begin(); it != gate_to_heat.end(); ++it )
        {
//            myOut << it->first << ",";
//            for( auto value : it->second )
//                myOut << value << ",";    
//            myOut << "\n";
            it->second.printGate( myOut, it->first );
        }
        cout<< "Finished project:" << project << endl;
    }
    
    cout << "Projects with erros:" << projectWithErrors.size() << endl;
    for( auto project : projectWithErrors )
        cout << project << endl;
    cout << endl;
    //show_rtree();
    return 0;
}

