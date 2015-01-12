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
#include <opm/core/simulator/SimulatorState.hpp>
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
    template <class DomainSpace, class RangeSpace = DomainSpace >
    class IstlMatrix : public DuneMatrix
    {
    public:
        typedef DuneMatrix  BaseType;
        typedef DomainSpace DomainSpaceType ;
        typedef RangeSpace  RangeSpaceType ;

#if HAVE_DUNE_FEM
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DomainSpaceType > RowDiscreteFunctionType ;
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< RangeSpaceType  > ColDiscreteFunctionType ;

        typedef typename RowDiscreteFunctionType :: DofStorageType  RowBlockVectorType;
        typedef typename ColDiscreteFunctionType :: DofStorageType  ColBlockVectorType;

        typedef typename RowDiscreteFunctionType :: GridType :: Traits ::
            CollectiveCommunication  CollectiveCommunictionType;
#endif

        IstlMatrix( const Eigen::SparseMatrix<double, Eigen::RowMajor>& matrix )
            : BaseType( matrix )
        {}
    };

    template <class GridImpl>
    class DuneFemGrid : public DuneGrid< GridImpl >
    {
        typedef DuneGrid< GridImpl > BaseType;
    public:
        typedef typename BaseType :: Grid Grid;
        typedef typename BaseType :: CollectiveCommunication  CollectiveCommunication;

        static const int dimension = Grid :: dimension ;

#if HAVE_DUNE_FEM
        typedef Dune::Fem::AdaptiveLeafGridPart< Grid >   AllGridPart;
        typedef typename AllGridPart :: GridViewType      AllGridView;
        typedef Dune::Fem::DGAdaptiveLeafGridPart< Grid > GridPart;
        typedef typename GridPart :: GridViewType         GridView;
#else
        typedef typename Grid :: LeafGridView             GridView;
        typedef GridView                                  AllGridView;
#endif

#if HAVE_DUNE_FEM
        typedef Dune::Fem::FunctionSpace< double, double, dimension, 1 >    FunctionSpace;
        typedef Dune::Fem::FiniteVolumeSpace< FunctionSpace, GridPart, /* codim = */ 0 >  FiniteVolumeSpace;
        typedef Dune::Fem::AdaptiveDiscreteFunction< FiniteVolumeSpace >    DiscreteFunction;

        typedef Dune::Fem::CombinedSpace< FiniteVolumeSpace, 3, Dune::Fem::VariableBased >  VectorSpaceType;

        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< FiniteVolumeSpace >  ISTLDiscreteFunction;
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< VectorSpaceType   >  ISTLVectorDiscreteFunction;

        // we can only deal with block size 1 at the moment
        static_assert( ISTLVectorDiscreteFunction::localBlockSize == 1, "blocksize error" );

        typedef IstlMatrix< VectorSpaceType   > SystemMatrixType;
        typedef IstlMatrix< FiniteVolumeSpace > EllipticMatrixType;

        typedef Dune::Fem::DGParallelMatrixAdapter< SystemMatrixType >   SystemMatrixAdapterType;
        typedef Dune::Fem::DGParallelMatrixAdapter< EllipticMatrixType > EllipticMatrixAdapterType;
#endif

        struct CreateGridPart
        {
            mutable std::unique_ptr< AllGridPart > gridPart_;
            AllGridView createGridView( Grid& grid ) const
            {
                if( ! gridPart_ )
                    gridPart_.reset( new AllGridPart( grid ) );
                return gridPart_->gridView();
            }
        };

        using BaseType :: grid_;
        using BaseType :: ug_;
        using BaseType :: grid;
        using BaseType :: cartDims_;
        using BaseType :: globalIndex_;

        DuneFemGrid(Opm::DeckConstPtr deck, const std::vector<double>& porv )
#if HAVE_DUNE_FEM
            : BaseType()
              grid_( createDuneGrid( deck, porv, CreateGridPart() ) ),
              allGridPart_( grid() ),
              gridPart_( grid() ),
              singleSpace_( gridPart_ ),
              vectorSpace_( gridPart_ )
#else
            : BaseType( deck, porv )
#endif
        {
#if HAVE_DUNE_FEM
            ug_.reset( dune2UnstructuredGrid( allGridPart_.gridView(), globalIndex(), cartDims_, true ) )
#endif
        }

#if HAVE_DUNE_FEM
        GridView gridView () const { return gridPart_.gridView(); }
#else
        GridView gridView () const { return grid().leafGridView(); }
#endif

#if HAVE_DUNE_FEM
        void communicate( SimulatorState& state ) const
        {
            if( singleSpace_.size() != state.pressure().size()  )
                std::cout << singleSpace_.size() << " " << state.pressure().size() << std::endl;
            assert( singleSpace_.size() == state.pressure().size() );
            DiscreteFunction p( "pressure", singleSpace_, &state.pressure()[0] );
            p.communicate();
            DiscreteFunction sat( "sat", singleSpace_, &state.saturation()[0] );
            sat.communicate();
            DiscreteFunction sat2( "sat2", singleSpace_, &state.saturation()[ singleSpace_.size() ] );
            sat2.communicate();
        }

        SystemMatrixAdapterType matrixAdapter( SystemMatrixType& matrix ) {
            typedef Dune::Fem::FemSeqILU0< SystemMatrixType, typename SystemMatrixType::RowBlockVectorType,
                                                             typename SystemMatrixType::ColBlockVectorType > PreconditionerType;
            typename SystemMatrixAdapterType::PreconditionAdapterType
                precon( matrix, 0, 1.0, (PreconditionerType *) 0 );
            return SystemMatrixAdapterType( matrix, vectorSpace_, vectorSpace_, precon );
        }

        void gather( const SimulatorState& localState )
        {
            // gather solution to rank 0 for EclipseWriter
        }
#endif

    protected:
#if HAVE_DUNE_FEM
        AllGridPart allGridPart_;
        GridPart gridPart_;
        FiniteVolumeSpace singleSpace_;
        VectorSpaceType   vectorSpace_;
#endif
    };
} // end namespace Opm
#endif
