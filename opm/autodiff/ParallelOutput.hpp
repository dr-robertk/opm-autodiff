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
#ifndef OPM_DUNEFEMGRID_HEADER_INCLUDED
#define OPM_DUNEFEMGRID_HEADER_INCLUDED

#include <opm/core/grid.h>
#include <opm/core/simulator/SimulatorState.hpp>
#include <opm/core/simulator/WellState.hpp>

#include <opm/autodiff/DuneMatrix.hpp>

// we need dune-cornerpoint for reading the Dune grid.
#include <opm/autodiff/DuneGrid.hpp>

#if HAVE_DUNE_ALUGRID
#include <dune/alugrid/grid.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#endif

#if HAVE_DUNE_FEM
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/space/finitevolume.hh>
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/function/blockvectorfunction.hh>
#include <dune/fem/function/combinedfunction.hh>
#include <dune/fem/operator/matrix/istlmatrixadapter.hh>
#include <dune/fem/operator/matrix/preconditionerwrapper.hh>
#else
#error "Need dune-fem for this header"
#endif

namespace Opm
{
    template <class GridImpl>
    class ParallelOutput : public DuneGrid< GridImpl >
    {
        typedef DuneGrid< GridImpl > BaseType;
    public:
        typedef typename BaseType :: Grid Grid;
        typedef typename BaseType :: CollectiveCommunication  CollectiveCommunication;

        static const int dimension = Grid :: dimension ;

        typedef typename Grid :: LeafGridView             GridView;
        typedef GridView                                  AllGridView;

#if HAVE_DUNE_ALUGRID
        typedef ALUGrid::MpAccessMPI   MpAccessType;
        typedef ALUGrid::ObjectStream  MessageBufferType;
#endif

        struct CreateGridView
        {
            CreateGridView() {}
            AllGridView operator ()( Grid& grid ) const
            {
                return grid.leafGridView();
            }
        };

        typedef std::vector<int>  IndexMapType;

        class DistributeLocalIds : public MpAccessType::NonBlockingExchange::DataHandleIF
        {
            const UnstructuredGrid& ug_;
            const int numInteriorCells_;

            std::vector< IndexMapType > indexMaps_;

            std::map< const int, const int > globalPosition_;
        public:
            DistributeLocalIds( const UnstructuredGrid& ug,
                                const int interiorCells,
                                std::vector< IndexMapType >& indexMaps,
                                std::unique_ptr< UnstructuredGrid >& globalUG )
            : ug_( ug ),
              numInteriorCells_( interiorCells ),
              indexMaps_( indexMaps ),
              globalPosition_()
            {
                if( globalUG )
                {
                    // insert position in grid iteration by global cell id
                    for ( int index = 0; index < globalUG->number_of_cells; ++index )
                    {
                        globalPosition_.insert( std::make_pair( globalUG->global_cell[ index ], index ) );
                    }
                }
            }

            void pack( const int link, MessageBufferType& buffer )
            {
                // we should only get one link
                assert( link == 0 );
                // pack all interior global cell id's
                buffer.write( numInteriorCells_ );
                for( int index = 0; index < numInteriorCells_; ++index )
                {
                    const int globalIdx = ug_.global_cell[ index ];
                    buffer.write( globalIdx );
                }
            }

            void unpack( const int link, MessageBufferType& buffer )
            {
                // assert( isIORank() );
                // get index map for current link
                IndexMapType& indexMap = indexMaps_[ link ];
                assert( ! globalPosition_.empty() );

                // unpack all interior global cell id's
                int numCells = 0;
                buffer.read( numCells );
                indexMap.resize( numCells );
                for( int index = 0; index < numCells; ++index )
                {
                    int globalId = -1;
                    buffer.read( globalId );
                    assert( globalPosition_.find( globalId ) != globalPosition_.end() );
                    indexMap[ index ] = globalPosition_[ globalId ];
                }
            }
        };

        UnstructuredGrid* createIOGrid()
        {
            UnstructuredGrid* globalUG = 0 ;
            if( isIORank() )
            {
                AllGridPart gridPart( grid() );
                std::cout << "Computing global unstructured grid for I/O" << std::endl;
                // create global unstructured grid
                globalUG = dune2UnstructuredGrid( gridPart.gridView(), globalIndex(), cartDims_, true );
            }

            // now distribute the grid (including globalIndex )
            distributeGrid( grid() );

            return globalUG;
        }


        using BaseType :: ug_;
        using BaseType :: grid;
        using BaseType :: cartDims_;
        using BaseType :: globalIndex;
        using BaseType :: createDuneGrid;
        using BaseType :: distributeGrid;
        using BaseType :: dune2UnstructuredGrid;
        using BaseType :: comm;

        DuneFemGrid(Opm::DeckConstPtr deck, const std::vector<double>& poreVolumes )
            : BaseType( createDuneGrid( deck, poreVolumes, CreateGridView(), false ) ),
              globalUG_( createIOGrid() ), // also distributes the grid
              toIORankComm_( comm() )
        {
            // create local unstructured grid
            ug_.reset( dune2UnstructuredGrid( grid().leafGridView(), globalIndex(), cartDims_, true ) );

            std::set< int > linkage;
            const int ioRank = 0 ;
            // rank 0 receives from all other ranks
            if( comm().rank() == ioRank )
            {
                for(int i=0; i<comm().size(); ++i)
                {
                    if( i != ioRank )
                    {
                        linkage.insert( MpAccessType::recvRank( i ) );
                    }
                }
            }
            else // all other simply send to rank 0
            {
                linkage.insert( MpAccessType::sendRank( ioRank ) );
            }

            // insert linkage to communicator
            toIORankComm_.insertRequestNonSymmetric( linkage );

            if( isIORank() )
            {
                // need an index map for each rank
                indexMaps_.resize( comm().size() );
            }

            // distribute global id's to io rank for later association of dof's
            DistributeLocalIds distIds( *ug_, ug_->number_of_cells, indexMaps_, globalUG_ );
            toIORankComm_.exchange( distIds );
        }

        GridView gridView () const { return grid().leafGridView(); }

        class PackUnPackSimulatorState : public MpAccessType::NonBlockingExchange::DataHandleIF
        {
            const SimulatorState& localState_;
            SimulatorState& globalState_;
            const WellState& localWellState_;
            WellState& globalWellState_;

            typedef std::vector<int>  IndexMapType;

            std::vector< IndexMapType > indexMap_;
        public:
            PackUnPackSimulatorState( const SimulatorState& localState,
                                      SimulatorState& globalState,
                                      const WellState& localWellState,
                                      WellState& globalWellState )
            : localState_( localState ),
              globalState_( globalState ),
              localWellState_( localWellState ),
              globalWellState_( globalWellState )
            {}

            template <class Vector>
            void write( MessageBufferType& buffer, const Vector& vector ) const
            {
                size_t size = vector.size();
                buffer.write( size );
                for( size_t i=0; i<size; ++i )
                    buffer.write( vector[i] );
            }

            template <class IndexMap, class Vector>
            void read( MessageBufferType& buffer,
                       const IndexMap& indexMap,
                       Vector& vector ) const
            {
                size_t size = 0;
                buffer.read( size );
                assert( size == indexMap.size() );
                for( size_t i=0; i<size; ++i )
                    buffer.read( vector[ indexMap[ i ] ] );
            }

            void pack( const int link, MessageBufferType& buffer )
            {
                // we should only get one link
                assert( link == 0 );
                // write all data from local state to buffer
                write( buffer, localState_.pressure()     );
                write( buffer, localState_.temperature()  );
                write( buffer, localState_.facepressure() );
                write( buffer, localState_.faceflux()     );
                write( buffer, localState_.saturation()   );

                // write all data from local well state to buffer
                /*
                write( buffer, localWellState_.bhp() );
                write( buffer, localWellState_.temperature() );
                write( buffer, localWellState_.wellRates() );
                write( buffer, localWellState_.perfRates() );
                write( buffer, localWellState_.perfPress() );
                */
            }

            void unpack( const int link, MessageBufferType& buffer )
            {
                assert( isIORank() );
                // get index map for current link
                const IndexMapType& indexMap = indexMap_[ link ];

                // write all data from local state to buffer
                read( buffer, indexMap, globalState_.pressure()     );
                read( buffer, indexMap, globalState_.temperature()  );
                read( buffer, indexMap, globalState_.facepressure() );
                read( buffer, indexMap, globalState_.faceflux()     );
                read( buffer, indexMap, globalState_.saturation()   );

                // TODO: well identification
                /*
                read( buffer, wellIndex, globalWellState_.bhp() );
                read( buffer, wellIndex, globalWellState_.temperature() );
                read( buffer, wellIndex, globalWellState_.wellRates() );
                read( buffer, wellIndex, globalWellState_.perfRates() );
                read( buffer, wellIndex, globalWellState_.perfPress() );
                */
            }
        };

        // gather solution to rank 0 for EclipseWriter
        void collectToIORank( SimulatorState& localState, const WellState& wellState )
        {
            communicate( localState );
            PackUnPackSimulatorState packUnpack( localState, globalState_,
                                                 wellState,  globalWellState_ );
            toIORankComm_.exchange( packUnpack );
        }

        bool isIORank() const
        {
            return comm().rank() == 0;
        }
#endif

    protected:
        std::unique_ptr< UnstructuredGrid > globalUG_;
#if HAVE_DUNE_FEM
        AllGridPart allGridPart_;
        GridPart gridPart_;
        FiniteVolumeSpace singleSpace_;
        VectorSpaceType   vectorSpace_;
        MpAccessType toIORankComm_;
        std::vector< std::vector< int > > indexMaps_;
#endif
        SimulatorState globalState_;
        WellState      globalWellState_;
    };

    namespace AutoDiffGrid
    {
        // derive from ADFaceCellTraits< UnstructuredGrid > since
        // DuneGrid casts into UnstructuredGrid
        template<class G>
        struct ADFaceCellTraits<DuneFemGrid< G > > : public ADFaceCellTraits< UnstructuredGrid >
        {
        };
    }

} // end namespace Opm
#endif
