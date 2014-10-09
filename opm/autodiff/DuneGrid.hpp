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

// we need dune-cornerpoint for reading the Dune grid.
#if HAVE_DUNE_CORNERPOINT
#include <dune/grid/CpGrid.hpp>
#else
#error This header needs the dune-cornerpoint module
#endif

namespace Opm
{
    template <class GridImpl>     
    class DuneGrid
    {
    public:
        typedef GridImpl Grid;

        static const int dimension = Grid :: dimension ;

        typedef typename Grid :: Traits :: template Codim< 0 > :: Entity                 Element;
        typedef typename Grid :: Traits :: template Codim< 0 > :: EntityPointer          ElementPointer;
        typedef typename Grid :: Traits :: template Codim< dimension > :: Entity         Vertex;
        typedef typename Grid :: Traits :: template Codim< dimension > :: EntityPointer  VertexPointer;

        typedef typename Element :: Geometry ElementGeometry ;
        typedef typename ElementGeometry :: GlobalCoordinate GlobalCoordinate;

        DuneGrid(Opm::DeckConstPtr deck, const std::vector<double>& porv )
        {
            if( porv.size() > 0 )
                OPM_THROW(std::runtime_error,"PORV not yet supported by DuneGrid");

            Dune::CpGrid cpgrid;
            // create CpGrid from deck
            cpgrid.processEclipseFormat(deck, 0.0, false, false);

            // grid factory converting a grid
            Dune::FromToGridFactory< Grid > factory;

            // create Grid from CpGrid
            Grid* grid = factory.convert( cpgrid );
            grid_.reset( grid );

            assert( grid_->size( 0 ) == cpgrid.numCells() );

            // create an UnstructuredGrid
            ug_.reset( dune2UnstructuredGrid( *grid_, true ) );

            for( int d=0; d<dimension; ++d )
                ug_->cartdims[ d ] = cpgrid.logicalCartesianSize()[ d ];

            assert( ug_->number_of_cells > 0 );
            if( ! ug_->global_cell )
                ug_->global_cell = (int *) std::malloc( ug_->number_of_cells * sizeof(int) );
            assert( int(cpgrid.globalCell().size()) == ug_->number_of_cells );
            // copy global cell information
            std::copy( cpgrid.globalCell().begin(), cpgrid.globalCell().end(), ug_->global_cell );
        }

        /// \brief destructor destroying the UnstructuredGrid
        ~DuneGrid()
        {
            UnstructuredGrid* ug = ug_.release();
            destroy_grid( ug );
        }

        // return Dune::Grid
        Grid& grid() { return *grid_; }

        // return unstructured grid
        UnstructuredGrid& c_grid() { return *ug_; }

        UnstructuredGrid* dune2UnstructuredGrid( Grid& grid, const bool faceTags )
        {
            typedef typename Grid :: ctype ctype;
            typedef typename Grid :: LeafGridView GridView ;
            typedef typename GridView :: template Codim< 0 > :: Iterator Iterator;
            typedef typename GridView :: IntersectionIterator      IntersectionIterator;
            typedef typename IntersectionIterator :: Intersection  Intersection;
            typedef typename Intersection :: Geometry              IntersectionGeometry;
            typedef typename GridView :: IndexSet                  IndexSet;

            GridView gridView = grid.leafGridView();
            const IndexSet& indexSet = gridView.indexSet();

            const int numCells = indexSet.size( 0 );
            const int numFaces = indexSet.size( 1 );
            const int numNodes = indexSet.size( dimension );

            const int maxNumVerticesPerFace = 4;
            const int maxNumFacesPerCell    = 6;

            // create Unstructured grid struct
            UnstructuredGrid* ug = allocate_grid( dimension, numCells, numFaces,
                    numFaces*maxNumVerticesPerFace,
                    numCells * maxNumFacesPerCell,
                    numNodes );

            int count = 0;
            int cellFace = 0;
            const Iterator end = gridView.template end<0> ();
            for( Iterator it = gridView.template begin<0> (); it != end; ++it, ++count )
            {
                const Element& element = *it;
                const ElementGeometry geometry = element.geometry();

                // currently only hexahedrons are supported
                // assert( element.type().isHexahedron() );

                const int index = indexSet.index( element );
                // make sure that the elements are ordered as before,
                // otherwise the globalCell mapping is invalid
                assert( count == index );

                const GlobalCoordinate center = geometry.center();
                int idx = index * dimension;
                for( int d=0; d<dimension; ++d, ++idx )
                    ug->cell_centroids[ idx ] = center[ d ];
                ug->cell_volumes[ index ] = geometry.volume();

                const int vertices = geometry.corners();
                for( int vx=0; vx<vertices; ++vx )
                {
                    const GlobalCoordinate vertex = geometry.corner( vx );
                    int idx = indexSet.subIndex( element, vx, dimension ) * dimension;
                    for( int d=0; d<dimension; ++d, ++idx )
                        ug->node_coordinates[ idx ] = vertex[ d ];
                }

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
                    int faceIdx = faceIndex * maxNumVerticesPerFace ;
                    ug->face_nodepos[ faceIndex   ] = faceIdx;
                    ug->face_nodepos[ faceIndex+1 ] = faceIdx + maxNumVerticesPerFace;
                    for( int vx=0; vx<vxSize; ++vx, ++faceIdx )
                    {
                        const int localVx = refElem.subEntity( localFace, 1, vx, dimension );
                        const int vxIndex = indexSet.subIndex( element, localVx, dimension );
                        ug->face_nodes[ faceIdx ] = vxIndex ;
                    }

                    assert( vxSize    <= maxNumVerticesPerFace );
                    assert( localFace <  maxNumFacesPerCell );

                    // store cell --> face relation
                    ug->cell_faces  [ cellFace + localFace ] = faceIndex;
                    if( faceTags )
                    {
                        // fill logical cartesian orientation of the face (here indexInInside)
                        ug->cell_facetag[ cellFace + localFace ] = localFace;
                    }

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
                    int idx = faceIndex * dimension;
                    for( int d=0; d<dimension; ++d, ++idx )
                    {
                        ug->face_normals  [ idx ] = normal[ d ];
                        ug->face_centroids[ idx ] = center[ d ];
                    }
                }
                if( faceCount > maxNumFacesPerCell )
                    OPM_THROW(std::logic_error,"DuneGrid only supports conforming hexahedral currently");
                cellFace += faceCount;
            }
            // set last entry
            ug->cell_facepos[ numCells ] = cellFace;

            if( cellFace < numFaces * maxNumFacesPerCell )
            {
                // reallocate data
                // ug->cell_faces
            }

            return ug;
        }

    protected:
        std::unique_ptr< Grid > grid_;
        std::unique_ptr< UnstructuredGrid > ug_;
    };
} // end namespace Opm
#endif
