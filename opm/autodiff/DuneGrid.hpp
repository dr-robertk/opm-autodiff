/*
  Copyright 2014 IRIS AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_DUNEGRID_HEADER_INCLUDED
#define OPM_DUNEGRID_HEADER_INCLUDED

#include <opm/core/grid.h>


#if HAVE_DUNE_ALUGRID
    #if HAVE_DUNE_CORNERPOINT
    #include <dune/grid/CpGrid.hpp>
    #else
    #error This header needs the dune-cornerpoint module
    #endif
#include <dune/alugrid/dgf.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#else
#error This header needs the dune-alugrid module
#endif

namespace Opm
{

    template <class G>     
    class DuneGrid
    {
    public:
        // hardcode ALUGrid for the moment
        typedef Dune :: ALUGrid< 3, 3, Dune::cube, Dune::nonconforming > Grid;

        static const int dimension = Grid :: dimension ;

        typedef typename Grid :: Traits :: template Codim< 0 > :: Entity                 Element;
        typedef typename Grid :: Traits :: template Codim< 0 > :: EntityPointer          ElementPointer;
        typedef typename Grid :: Traits :: template Codim< dimension > :: Entity         Vertex;
        typedef typename Grid :: Traits :: template Codim< dimension > :: EntityPointer  VertexPointer;

        typedef typename Element :: Geometry ElementGeometry ;
        typedef typename ElementGeometry :: GlobalCoordinate GlobalCoordinate;

        DuneGrid(Opm::DeckConstPtr deck) 
        {
            Dune::CpGrid cpgrid;
            // create CpGrid from deck
            cpgrid.processEclipseFormat(deck, 0.0, false, false);

            // grid factory converting a grid
            Dune::FromToGridFactory< Grid > factory;

            // create Grid from CpGrid
            Grid* grid = factory.convert( cpgrid );
            grid_.reset( grid );

            assert( grid_->size( 0 ) == cpgrid.numCells() );

            ug_.reset( init( *grid_ ) );
            for( int d=0; d<dimension; ++d )
                ug_->cartdims[ d ] = cpgrid.logicalCartesianSize()[ d ];

            assert( ug_->number_of_cells > 0 );
            if( ! ug_->global_cell ) 
                ug_->global_cell = (int *) std::malloc( ug_->number_of_cells * sizeof(int) );
            assert( int(cpgrid.globalCell().size()) == ug_->number_of_cells );
            // copy global cell information
            std::copy( cpgrid.globalCell().begin(), cpgrid.globalCell().end(), ug_->global_cell );

            std::exit( 0 );
        }

        // return Dune::Grid
        Grid& grid() { return *grid_; }

        // return unstructured grid 
        UnstructuredGrid& c_grid() { return *ug_; }

        UnstructuredGrid* init( Grid& grid ) 
        {
            typedef typename Grid :: ctype ctype;
            typedef typename Grid :: LeafGridView GridView ;
            typedef typename GridView :: template Codim< 0 > :: Iterator Iterator;
            typedef typename GridView :: IntersectionIterator      IntersectionIterator;
            typedef typename IntersectionIterator :: Intersection  Intersection;
            typedef typename Intersection :: Geometry             IntersectionGeometry;
            typedef typename GridView :: IndexSet                 IndexSet;

            GridView gridView = grid.leafGridView();
            const IndexSet& indexSet = gridView.indexSet();

            const int numCells = indexSet.size( 0 );
            const int numFaces = indexSet.size( 1 );
            const int numNodes = indexSet.size( dimension );
            const int numVerticesPerFace = 4;
            const int numFacesPerCell    = 6;

            // create Unstructured grid struct
            UnstructuredGrid* ug = allocate_grid( dimension, numCells, numFaces,
                    numFaces*numVerticesPerFace, numCells*numFacesPerCell, numNodes ); 

            int count = 0;
            const Iterator end = gridView.template end<0> ();
            for( Iterator it = gridView.template begin<0> (); it != end; ++it, ++count )
            {
                const Element& element = *it ;
                const ElementGeometry geometry = element.geometry();
                const int index = indexSet.index( element );
                // make sure that the elements are ordered as before, 
                // otherwise the globalCell mapping is invalid
                assert( count == index ); 

                const GlobalCoordinate center = geometry.center();
                int idx = index * dimension ;
                for( int d=0; d<dimension; ++d, ++idx )
                    ug->cell_centroids[ idx ] = center[ d ];
                ug->cell_volumes[ index ] = geometry.volume();
                
                const int vertices = geometry.corners();
                for( int vx=0; vx<vertices; ++vx ) 
                {
                    const GlobalCoordinate vertex = geometry.corner( vx );
                    int idx = indexSet.subIndex( element, vx, dimension ) * dimension ;
                    for( int d=0; d<dimension; ++d, ++idx )
                        ug->node_coordinates[ idx ] = vertex[ d ];
                }

                const int cellFace = numFacesPerCell * indexSet.index( element );
                ug->cell_facepos[ index ] = cellFace;

                const Dune::ReferenceElement< ctype, dimension > &refElem
                        = Dune::ReferenceElements< ctype, dimension >::general( element.type() );

                int faceCount = 0;
                const IntersectionIterator endiit = gridView.iend( element );
                for( IntersectionIterator iit = gridView.ibegin( element ); iit != endiit; ++iit, ++faceCount )
                {
                    const Intersection& intersection = *iit;
                    IntersectionGeometry intersectionGeometry = intersection.geometry();
                    const double faceVol = intersectionGeometry.volume();

                    const int localFace = intersection.indexInInside();
                    const int faceIndex = indexSet.subIndex( element, localFace, 1 );

                    ug->face_areas[ faceIndex ] = faceVol;

                    // get number of vertices (should be 4)
                    const int vxSize = refElem.size( localFace, 1, dimension );
                    int faceIdx = faceIndex * numVerticesPerFace ;
                    for( int vx=0; vx<vxSize; ++vx, ++faceIdx )
                    {
                        const int localVx = refElem.subEntity( localFace, 1, vx, dimension );
                        const int vxIndex = indexSet.subIndex( element, localVx, dimension );
                        ug->face_nodes[ faceIdx ] = vxIndex ;
                    }

                    assert( vxSize == numVerticesPerFace );
                    assert( localFace < numFacesPerCell );

                    // store cell --> face relation
                    ug->cell_faces[ cellFace + localFace ] = faceIndex;

                    GlobalCoordinate normal = intersection.centerUnitOuterNormal();   
                    normal *= faceVol;

                    // store face --> cell relation
                    if( intersection.neighbor() )
                    {
                        ElementPointer ep = intersection.outside();
                        const Element& neighbor = *ep;
                        const int nbIndex = indexSet.index( neighbor );
                        if( index < nbIndex ) 
                        {
                            ug->face_cells[ 2*faceIndex     ] = index;
                            ug->face_cells[ 2*faceIndex + 1 ] = nbIndex;
                        }
                        else 
                        {
                            ug->face_cells[ 2*faceIndex     ] = nbIndex;
                            ug->face_cells[ 2*faceIndex + 1 ] = index;
                            // flip normal
                            normal *= -1.0;
                        }
                    }
                    else 
                    {
                        ug->face_cells[ 2*faceIndex     ] = index;
                        ug->face_cells[ 2*faceIndex + 1 ] = -1; // boundary
                    }

                    const GlobalCoordinate center = intersectionGeometry.center();
                    // store normal
                    int idx = faceIndex * dimension ;
                    for( int d=0; d<dimension; ++d, ++idx )
                    {
                        ug->face_normals  [ idx ] = normal[ d ];
                        ug->face_centroids[ idx ] = center[ d ];
                    }
                }
                if( faceCount != numFacesPerCell ) 
                    OPM_THROW(std::logic_error,"DuneGrid only supports conforming hexahedral currently");
            }
            return ug;
        }

    protected:        
        std::unique_ptr< Grid > grid_; 
        std::unique_ptr< UnstructuredGrid > ug_;
    };
} // end namespace Opm
#endif
